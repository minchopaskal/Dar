#pragma once

#include <Windows.h>
#include <comdef.h> // _com_error

#include "utils/defines.h"

template <class Mutex>
struct Lock {
	~Lock() {
		if (mutex) {
			mutex->unlock();
		}
	}

	bool locked() const {
		return mutex != nullptr;
	}

private:
	Lock(Mutex *mutex) : mutex(mutex) {}

	Mutex *mutex;

	friend struct CriticalSection;
	friend struct SpinLock;
};

struct CriticalSection {
	using LockType = Lock<CriticalSection>;

	CriticalSection(bool inited = false) : cs{} {
		this->inited = false;
		if (inited) {
			init();
		}
	}

	~CriticalSection() {
		if (inited) {
			DeleteCriticalSection(&cs);
		}
	}

	bool init() {
		if (inited) {
			return true;
		}

		const int spinCount = 100; // TODO: maybe make a parameter or see which is the best value
		BOOL res;
#ifdef DAR_DEBUG
		res = InitializeCriticalSectionEx(&cs, spinCount, 0);
		if (!res) {
			_com_error err(GetLastError());
			OutputDebugString(err.ErrorMessage());
		}
#else
		res = InitializeCriticalSectionEx(&cs, spinCount, CRITICAL_SECTION_NO_DEBUG_INFO);
#endif // DAR_DEBUG
	
		if (res) {
			inited = true;
		}

		return res;
	}

	[[nodiscard]] LockType lock() {
		if (inited) {
			EnterCriticalSection(&cs);
		}

		return LockType{ inited ? this : nullptr };
	}

	[[nodiscard]] LockType tryLock() {
		if (inited && TryEnterCriticalSection(&cs)) {
			return LockType{ this };
		} else {
			return LockType{ nullptr };
		}
	}

private:
	__forceinline void unlock() {
		dassert(inited); // Only the Lock structure should call this method.
		LeaveCriticalSection(&cs);
	}

private:
	CRITICAL_SECTION cs;
	bool inited;

	friend struct LockType;
};

struct Event {
	Event(bool manualReset = false) {
		h = CreateEvent(
			NULL,
			manualReset,
			false,
			nullptr
		);
	}

	~Event() {
		reset();
		CloseHandle(h);
	}

	void signal() {
		SetEvent(h);
	}

	/// @return true if the event was signaled, false otherwise.
	bool wait(unsigned int millis = 0) {
		DWORD res = WaitForSingleObject(h, millis == 0 ? INFINITE : millis);
		return (res == WAIT_OBJECT_0);
	}

	/// Set the event in a non-signalled state.
	void reset() {
		ResetEvent(h);
	}

private:
	HANDLE h = NULL;
};

// Simple spin lock type.
struct SpinLock {
	using LockType = Lock<SpinLock>;

	SpinLock() : locked(0) { }

	[[nodiscard]] LockType lock() {
		while (true) {
			while (locked.load() != 0) {
				YieldProcessor();
				YieldProcessor();
			}

			int expected = 0;
			if (locked.compare_exchange_strong(expected, 1)) {
				break;
			}
		}

		return LockType{ this };
	}

	[[nodiscard]] LockType tryLock() {
		int expected = 0;
		if (locked.compare_exchange_strong(expected, 1)) {
			return LockType{ this };
		}
		return LockType{ nullptr };
	}

private:
	void unlock() {
		int expected = 1;
		bool res = locked.compare_exchange_strong(expected, 0);
		
		// Only the Lock<SpinLock> destructor should be able to call the unlock method.
		// Thus the spin lock should have been locked.
		dassert(res);
	}

private:
	Atomic<int> locked;

	friend struct LockType;
};

/// @typeparam T Type of the objects which will be stored in the queue.
/// @typeparam N Size of the queue.
/// @typeparam eventful If true one can wait on an event the queue sets when a new value's pushed into it.
template <class T, int SIZE, bool EVENTFUL = false>
struct ThreadSafeQueue {
	ThreadSafeQueue() {}

	bool push(const T &val) {
		auto lock = cs.lock();
		if (full()) {
			return false;
		}

		buffer[tail] = val;
		tail = (tail + 1) % SIZE;

		if (EVENTFUL) {
			dataPushed.signal();
		}

		return true;
	}

	bool pop(T &res) {
		auto lock = cs.lock();
		if (empty()) {
			return false;
		}

		res = buffer[head];
		head = (head + 1) % SIZE;

		if (EVENTFUL && empty()) {
			dataPushed.reset();
		}

		return true;
	}

	bool empty() const {
		return head == tail;
	}

	bool full() const {
		return ((tail + 1) % SIZE) == head;
	}

	bool waitForData(unsigned int millis) {
		if (!EVENTFUL) {
			return !empty();
		}
		return dataPushed.wait(millis);
	}

private:
	T buffer[SIZE] = {};
	SpinLock cs;
	int head = 0;
	int tail = 0;
	Event dataPushed;
};
