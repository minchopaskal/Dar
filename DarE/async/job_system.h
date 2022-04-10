#pragma once

#include "utils/defines.h"

namespace Dar {

namespace JobSystem {

using JobFunction = void(*)(void*);

enum class JobType {
	Default = 0, // Generic work, any thread my take the work.
	Windows, // This type of job could be only ran on a specific thread. F.e input handling.

	Count
};

struct JobDecl {
	JobFunction f = nullptr;
	void *param = nullptr;
};

struct Fence;

/// Intialize the job system
void init();

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

} // JobSystem

} // namespace Dar