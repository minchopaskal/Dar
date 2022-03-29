#pragma once

#include <Windows.h>
#include <comdef.h> // _com_error

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
	CriticalSection(bool inited = false) : cs{}, inited(inited) {
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
			inited = true;
		}

		return res;
	}

	CriticalSectionLock lock() {
		if (inited) {
			EnterCriticalSection(&cs);
		}
		return CriticalSectionLock{ inited ? &cs : nullptr };
	}

	CriticalSectionLock tryLock() {
		if (inited && TryEnterCriticalSection(&cs)) {
			return CriticalSectionLock{ &cs };
		} else {
			return CriticalSectionLock{ nullptr };
		}
	}

private:
	CRITICAL_SECTION cs;
	bool inited;
};