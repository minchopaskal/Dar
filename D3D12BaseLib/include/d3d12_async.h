#pragma once

#include <synchapi.h>
#include <comdef.h>

struct CriticalSectionLock {
	CriticalSectionLock(CRITICAL_SECTION *cs) : cs(cs) { }
	~CriticalSectionLock() {
		if (cs) {
			LeaveCriticalSection(cs);
		}
	}

private:
	CRITICAL_SECTION *cs;
};

struct CriticalSection {
	CriticalSection(bool init = false) : init(init) {
		if (init) {
			initialize();
		}
	}

	~CriticalSection() {
		DeleteCriticalSection(&cs);
	}

	bool initialize() {
		if (init) {
			return true;
		}

		const int spinCount = 100; // TODO: maybe make a parameter or see which is the best value
		BOOL res;
#ifdef D3D12_DEBUG
		res = InitializeCriticalSectionEx(&cs, spinCount, 0);
		if (!res) {
			_com_error err(GetLastError());
			OutputDebugString(err.ErrorMessage());
		}
#else
		res = InitializeCriticalSectionEx(&cs, spinCount, CRITICAL_SECTION_NO_DEBUG_INFO);
#endif // D3D12_DEBUG
	
		if (res) {
			init = true;
		}

		return res;
	}

	CriticalSectionLock lock() {
		if (init) {
			EnterCriticalSection(&cs);
		}
		return CriticalSectionLock{ init ? &cs : nullptr };
	}

	CriticalSectionLock tryLock() {
		if (init && TryEnterCriticalSection(&cs)) {
			return CriticalSectionLock{ &cs };
		} else {
			return CriticalSectionLock{ nullptr };
		}
	}

private:
	CRITICAL_SECTION cs;
	bool init;
};