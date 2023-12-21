#include "job_system.h"

#include "async/async.h"
#include "math/dar_math.h"
#include "utils/profile.h"

#include <thread>

using namespace Dar;

/// ==========================================================================
/// Implementation
/// ==========================================================================

constexpr SizeType NUM_FIBERS = 160;
using FiberHandle = SizeType;
constexpr FiberHandle INVALID_FIBER_HANDLE = SizeType(-1);
Map<JobSystem::Fence*, ThreadSafeQueue<FiberHandle, NUM_FIBERS>> waitingFibers;
ThreadSafeQueue<FiberHandle, NUM_FIBERS> waitingReadyDefaultFibers;
ThreadSafeQueue<FiberHandle, NUM_FIBERS> waitingReadyWindowsFibers;
ThreadSafeQueue<FiberHandle, NUM_FIBERS> waitingReadyNonWindowsFibers;
SpinLock waitingFibersCS;
SpinLock waitingDefaultFibersCS;
SpinLock waitingWindowsFibersCS;
SpinLock waitingNonWindowsFibersCS;
uint32_t numThreads;

struct JobSystem::Fence {
	Fence() {}

	void init(int numJobs) {
		value += numJobs;
	}

	void decrement() {
		value.fetch_sub(1, std::memory_order_relaxed);
	}

	bool ready() const{
		return value.load() == 0;
	}

	void addWaitingFiber(FiberHandle handle) {
		auto lock = waitingFibersCS.lock();
		waitingFibers[this].push(handle);
	}

private:
	Atomic<int> value = 0;
};

struct Job {
	JobSystem::JobFunction function = nullptr;
	void *param = nullptr;
	JobSystem::Fence *fence = nullptr;
	JobSystem::JobType type = JobSystem::JobType::Default;
};

struct Fiber {
	LPVOID address = nullptr; // address of the fiber
	LPVOID executionThread = nullptr; // Address of the fiber of one of the main threads that switched to this fiber.
	UINT executionThreadIndex = UINT(-1); // Index of the execution thread
	JobSystem::JobType currentJobType = JobSystem::JobType::Default; // Type of the job the fiber started.
};

/// ==========================================================================
///  Globals
/// ==========================================================================
constexpr int WINDOWS_THREAD_INDEX = 0;
LPVOID WINDOWS_THREAD_FIBER = nullptr;

ThreadSafeQueue<Job, 1000/*size*/, true/*eventful*/> defaultJobsQueue;
ThreadSafeQueue<Job, 100/*size*/, true/*eventful*/> nonWindowsJobsQueue;
ThreadSafeQueue<Job, 100/*size*/, true/*eventful*/> windowsJobsQueue;
Vector<HANDLE> threads;
Fiber fibers[NUM_FIBERS];
ThreadSafeQueue<FiberHandle, NUM_FIBERS> fibersPool;
Atomic<int> stopJobSystem;
UINT fiberToThreadIndex[NUM_FIBERS];

Fiber& getFiberFromHandle(FiberHandle handle) {
	return fibers[handle];
}

void switchToFiberHandle(FiberHandle handle) {
	Fiber &f = getFiberFromHandle(handle);
	SwitchToFiber(f.address);
}

FiberHandle getFreeFiber() {
	FiberHandle index = INVALID_FIBER_HANDLE;
	while (!fibersPool.pop(index)) { }

	return index;
}

FiberHandle getCurrentFiberHandle() {
	PVOID addr = GetCurrentFiber();

	for (int i = 0; i < NUM_FIBERS; ++i) {
		if (fibers[i].address == addr) {
			return i;
		}
	}

	dassert(false);

	return INVALID_FIBER_HANDLE;
}

DWORD jobExecutionThread(void *param) {
	auto threadIndex = static_cast<uint32_t>(reinterpret_cast<SizeType>(param));
	char threadName[16];
	sprintf(threadName, "Thread[%du]", threadIndex);
	
	DAR_OPTICK_THREAD(threadName);

	LPVOID currentFiberAddress = ConvertThreadToFiber(nullptr);
	if (threadIndex == WINDOWS_THREAD_INDEX) {
		WINDOWS_THREAD_FIBER = currentFiberAddress;
	}

	while (stopJobSystem.load() == 0) {
		FiberHandle handle = getFreeFiber();

		if (handle == INVALID_FIBER_HANDLE) {
			continue;
		}

		Fiber &f = getFiberFromHandle(handle);
		f.executionThread = currentFiberAddress;
		f.executionThreadIndex = threadIndex;

		SwitchToFiber(f.address);
	}

	return ConvertFiberToThread();
}

FiberHandle searchWaiting(bool isWindows) {
	auto &waitingQueue = isWindows ? waitingReadyWindowsFibers : waitingReadyDefaultFibers;
	auto &waitingQueueCS = isWindows ? waitingWindowsFibersCS : waitingDefaultFibersCS;

	FiberHandle handle = INVALID_FIBER_HANDLE;
	{
		auto lock = waitingQueueCS.lock();
		if (!waitingQueue.empty()) {
			waitingQueue.pop(handle);
		}
	}

	if (handle == INVALID_FIBER_HANDLE) {
		if (isWindows) {
			auto lock = waitingDefaultFibersCS.lock();
			if (!waitingReadyDefaultFibers.empty()) {
				waitingReadyDefaultFibers.pop(handle);
			}
		} else {
			auto lock = waitingNonWindowsFibersCS.lock();
			if (!waitingReadyNonWindowsFibers.empty()) {
				waitingReadyNonWindowsFibers.pop(handle);
			}
		}
	}

	return handle;
}

void fiberStartRoutine(void *param) {
	FiberHandle fiberIndex = reinterpret_cast<FiberHandle>(param);
	Fiber &thisFiber = getFiberFromHandle(fiberIndex);

	if (thisFiber.executionThread == nullptr || thisFiber.executionThreadIndex == UINT(-1)) {
		dassert(false);
		exit(1);
	}

	while (stopJobSystem.load() == 0) {
		bool isWindowsFiber = thisFiber.executionThread == WINDOWS_THREAD_FIBER;
		FiberHandle waitingHandle = searchWaiting(isWindowsFiber);
		
		if (waitingHandle != INVALID_FIBER_HANDLE) {
			Fiber &f = getFiberFromHandle(waitingHandle);
			f.executionThread = thisFiber.executionThread;
			f.executionThreadIndex = thisFiber.executionThreadIndex;

			fibersPool.push(fiberIndex);
			SwitchToFiber(f.address);
		}

		Job job;
		bool jobFound = false;
		if (isWindowsFiber) {
			jobFound = windowsJobsQueue.pop(job);
		}

		if (!jobFound) {
			// Prioritize non-windows jobs for non-main threads
			if (!isWindowsFiber) {
				jobFound = nonWindowsJobsQueue.pop(job);
			} else {
				jobFound = defaultJobsQueue.pop(job);
			}
		}

		if (!jobFound) {
			if (!isWindowsFiber) {
				if (!defaultJobsQueue.pop(job)) {
					continue;
				}
			} else {
				continue;
			}
		}

		if (job.function == nullptr) {
			continue;
		}

		thisFiber.currentJobType = job.type;

		job.function(job.param);
		if (job.fence != nullptr) { // if no one is waiting on the job. Nothing to do anymore.
			job.fence->decrement();

			// If the fence completed make the fibers waiting on
			// it ready for execution.
			if (job.fence->ready()) {
				auto lock = waitingFibersCS.lock();
				auto it = waitingFibers.find(job.fence);
				if (it != waitingFibers.end()) {
					auto &waitingFibersQueue = it->second;
					FiberHandle handle;
					while (waitingFibersQueue.pop(handle)) {
						Fiber &f = getFiberFromHandle(handle);
						switch (f.currentJobType) {
						case JobSystem::JobType::Default:
							waitingReadyDefaultFibers.push(handle);
							break;
						case JobSystem::JobType::NonWindows:
							waitingReadyNonWindowsFibers.push(handle);
							break;
						case JobSystem::JobType::Windows:
							waitingReadyWindowsFibers.push(handle);
							break;
						}
					}
				}
			}
		}

		fibersPool.push(fiberIndex);
		SwitchToFiber(thisFiber.executionThread);
	}

	SwitchToFiber(thisFiber.executionThread);
}

JobSystem::Fence *getFreeFence() {
	// TODO: allocate memory for fences beforehand
	return new JobSystem::Fence();
}

/// ==========================================================================
/// Public
/// ==========================================================================
int JobSystem::init(int nt) {
	const uint32_t numSystemThreads = std::min(std::max(1u, std::thread::hardware_concurrency()), 64u);

	numThreads = nt > 0 ? std::min(static_cast<uint32_t>(nt), numSystemThreads) : numSystemThreads;

	threads.resize(numThreads);

	for (SizeType i = 0; i < NUM_FIBERS; ++i) {
		fibers[i].address = CreateFiber(
			64 * 1024,
			fiberStartRoutine,
			reinterpret_cast<void*>(i)
		);

		fibersPool.push(i);
	}

	stopJobSystem = 0;

	// Create worker fibers
	for (uint32_t i = 0; i < numThreads; ++i) {
		threads[i] = CreateThread(
			NULL,
			0,
			jobExecutionThread,
			reinterpret_cast<void*>(static_cast<SizeType>(i)),
			CREATE_SUSPENDED, // Do not start the thread until we've set its affinity
			NULL
		);

		unsigned int affinityMask = (1 << i);
		if (!SetThreadAffinityMask(threads[i], affinityMask)) {
			LOG(Error, "Failed to set affinity of created thread!");

			dassert(false);

			// Just exit, nothing to do anymore
			exit(1);
		}

		ResumeThread(threads[i]);
	}

	return numThreads;
}

void JobSystem::stop() {
	++stopJobSystem;
}

void JobSystem::kickJobs(JobSystem::JobDecl *jobs, int numJobs, JobSystem::Fence **fence, JobSystem::JobType type) {
	if (fence != nullptr) {
		if (*fence == nullptr) {
			*fence = getFreeFence();
		}

		(*fence)->init(numJobs);
	}

	for (int i = 0; i < numJobs; ++i) {
		Job job = { jobs[i].f, jobs[i].param, fence ? *fence : nullptr, type };
		switch (type) {
		case JobType::Default:
			defaultJobsQueue.push(job);
			break;
		case JobType::NonWindows:
			nonWindowsJobsQueue.push(job);
			break;
		case JobType::Windows:
			windowsJobsQueue.push(job);
			break;
		}
	}
}

void JobSystem::kickJobsAndWait(JobSystem::JobDecl *jobs, int numJobs, JobSystem::JobType type) {
	JobSystem::Fence *f = nullptr;

	kickJobs(jobs, numJobs, &f, type);

	waitFenceAndFree(f);
}

void JobSystem::waitFence(JobSystem::Fence *fence) {
	if (fence == nullptr) {
		return;
	}

	if (fence->ready()) {
		return;
	}

	FiberHandle handle = getCurrentFiberHandle();
	if (handle != INVALID_FIBER_HANDLE) {
		fence->addWaitingFiber(handle);
	}

	SwitchToFiber(getFiberFromHandle(handle).executionThread);
}

void JobSystem::waitFenceAndFree(JobSystem::Fence *&fence) {
	waitFence(fence);

	delete fence;
	fence = nullptr;
}

void JobSystem::waitForAll() {
	for (auto &t : threads) {
		WaitForSingleObject(t, INFINITE);
	}

	for (auto &f : fibers) {
		DeleteFiber(f.address);
		f.address = nullptr;
	}
}

bool JobSystem::probeFence(Fence *fence) {
	return !fence || fence->ready();
}

UINT JobSystem::getCurrentThreadIndex() {
	return fibers[getCurrentFiberHandle()].executionThreadIndex;
}

int JobSystem::getNumThreads() {
	return numThreads;
}
