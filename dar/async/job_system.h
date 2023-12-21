#pragma once

#include "utils/defines.h"

namespace Dar {

namespace JobSystem {

using JobFunction = void(*)(void*);

enum class JobType {
	Default = 0, ///< Generic work, any thread my take the work.
	NonWindows, ///< Work that must not be ran on the main thead. F.e when we want to ensure
	///< specific job is ran asynchronosly to the main thread.
	Windows, // This type of job could be only ran on the main thread. F.e useful for input handling.

	Count
};

struct JobDecl {
	JobFunction f = nullptr;
	void *param = nullptr;
};

struct Fence;

UINT getCurrentThreadIndex();

int getNumThreads();

/// Intialize the job system
/// @param numThreads Set desired number of threads used for the process.
///                   If <= 0 sets to maximum number of threads.
/// @return number of threads the job system was initialized with
int init(int numThreads);

/// Stop the job system
void stop();

/// Kick a batch of jobs. If the passed fence is valid one can
/// wait for the completion of the jobs by calling
/// waitForFence(AndFree) on the fence.
void kickJobs(JobDecl *jobs, int numJobs, Fence **fence, JobType type = JobType::Default);

/// Kick a batch of jobs and wait for their completion.
void kickJobsAndWait(JobDecl *jobs, int numJobs, JobType type = JobType::Default);

/// If the given fence is valid wait for the jobs associated with it
/// to complete.
void waitFence(Fence *fence);

/// If the given fence is valid wait for the jobs associated with it
/// to complete. After that release it.
void waitFenceAndFree(Fence *&fence);

/// Wait for all processing to stop.
void waitForAll();

/// @brief If the fence is valid check if it's ready without giving up execution control.
/// @return if the fence is invalid - true, otherwise - if it's ready.
bool probeFence(Fence *fence);

} // JobSystem

} // namespace Dar