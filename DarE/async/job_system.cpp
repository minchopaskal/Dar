#include "job_system.h"

#include "async/async.h"
#include "math/dar_math.h"

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
SpinLock waitingFibersCS;
SpinLock waitingDefaultFibersCS;
SpinLock waitingWindowsFibersCS;

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
	JobSystem::JobType currentJobType = JobSystem::JobType::Default; // Type of the job the fiber started.
};

/// ==========================================================================
///  Globals
/// ==========================================================================
constexpr int WINDOWS_THREAD_INDEX = 0;
LPVOID WINDOWS_THREAD_FIBER = nullptr;

ThreadSafeQueue<Job, 1000/*size*/, true/*eventful*/> defaultJobsQueue;
ThreadSafeQueue<Job, 100/*size*/, true/*eventful*/> windowsJobsQueue;
Vector<HANDLE> threads;
Fiber fibers[NUM_FIBERS];
ThreadSafeQueue<FiberHandle, NUM_FIBERS> fibersPool;
Atomic<int> stopJobSystem;

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
	SizeType threadIndex = reinterpret_cast<SizeType>(param);

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

	return handle;
}

void fiberStartRoutine(void *param) {
	FiberHandle fiberIndex = reinterpret_cast<FiberHandle>(param);
	Fiber &thisFiber = getFiberFromHandle(fiberIndex);

	if (thisFiber.executionThread == nullptr) {
		dassert(false);
		exit(1);
	}

	while (stopJobSystem.load() == 0) {
		bool isWindowsFiber = thisFiber.executionThread == WINDOWS_THREAD_FIBER;
		FiberHandle waitingHandle = searchWaiting(isWindowsFiber);
		if (waitingHandle == INVALID_FIBER_HANDLE && isWindowsFiber) {
			FiberHandle waitingHandle = searchWaiting(false);
		}

		if (waitingHandle != INVALID_FIBER_HANDLE) {
			Fiber &f = getFiberFromHandle(waitingHandle);
			f.executionThread = thisFiber.executionThread;

			fibersPool.push(fiberIndex);
			SwitchToFiber(f.address);
		}

		Job job;
		bool jobFound = false;
		if (isWindowsFiber) {
			jobFound = windowsJobsQueue.pop(job);
		}

		if (!jobFound && !defaultJobsQueue.pop(job)) {
			continue;
		}

		if (job.function == nullptr) {
			continue;
		}

		thisFiber.currentJobType = job.type;

		job.function(job.param);
		if (job.fence != nullptr) { // no one is waiting on the job. Nothing to do anymore.
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
						if (f.currentJobType == JobSystem::JobType::Windows) {
							waitingReadyWindowsFibers.push(handle);
						} else {
							waitingReadyDefaultFibers.push(handle);
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
void JobSystem::init() {
	stopJobSystem = 0;

	const unsigned int numSystemThreads = dmath::min(dmath::max(1u, std::thread::hardware_concurrency()), 64u);

	threads.resize(numSystemThreads);

	for (SizeType i = 0; i < NUM_FIBERS; ++i) {
		fibers[i].address = CreateFiber(
			64 * 1024,
			fiberStartRoutine,
			reinterpret_cast<void*>(i)
		);

		fibersPool.push(i);
	}

	// Create worker fibers
	for (SizeType i = 0; i < numSystemThreads; ++i) {
		threads[i] = CreateThread(
			NULL,
			0,
			jobExecutionThread,
			reinterpret_cast<void *>(i),
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
}

void Dar::JobSystem::stop() {
	++stopJobSystem;
}

void JobSystem::kickJobs(JobSystem::JobDecl *jobs, int numJobs, JobSystem::Fence **fence, JobSystem::JobType type) {
	if (fence != nullptr) {
		*fence = getFreeFence();
		(*fence)->init(numJobs);
	}

	for (int i = 0; i < numJobs; ++i) {
		Job job = { jobs[i].f, jobs[i].param, fence ? *fence : nullptr, type };
		if (type == JobSystem::JobType::Windows) {
			windowsJobsQueue.push(job);
		} else {
			defaultJobsQueue.push(job);
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
	}
}
