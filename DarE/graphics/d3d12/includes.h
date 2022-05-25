#pragma once

#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>

#include "AgilitySDK/include/d3d12.h"
#include <dxgi.h>
#include <dxgi1_6.h>

#ifdef DAR_DEBUG
#include "dxgidebug.h"
#endif // DAR_DEBUG

#include <wrl/client.h>
using Microsoft::WRL::ComPtr;

