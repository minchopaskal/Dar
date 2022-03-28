#include "d3d12_logger.h"
#include <cstdio>

#define OUT_STREAM stdout
#define IN_STREAM stdin
#define ERR_STREAM stderr

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\u001b[34;1m"
#define ANSI_COLOR_RESET   "\x1b[0m"

void Dar::Logger::setLogLevel(LogLevel lvl) {
	loggingLevel = static_cast<int>(lvl);
}

void Dar::Logger::log(LogLevel lvl, const char *fmt, ...) {
#ifndef CUDA_DEBUG
	if (lvl == LogLevel::Debug) {
		return;
	}
#endif

	int currentLvl = static_cast<int>(lvl);
	if (lvl != LogLevel::Debug && lvl != LogLevel::Error && currentLvl > loggingLevel) {
		return;
	}

	const char *color = nullptr;
	int levelIdx = 0;
	switch (lvl) {
	case LogLevel::Info:
		color = ANSI_COLOR_RESET;
		break;
	case LogLevel::InfoFancy:
		color = ANSI_COLOR_BLUE;
		break;
	case LogLevel::Warning:
		color = ANSI_COLOR_YELLOW;
		break;
	case LogLevel::Error:
		color = ANSI_COLOR_RED;
		break;
	case LogLevel::Debug:
		color = ANSI_COLOR_GREEN;
		break;
	default:
		levelIdx = -1;
		color = ANSI_COLOR_RESET;
		break;
	}

	const int infoIdx = static_cast<int>(LogLevel::Info);
	levelIdx = levelIdx < 0 ? infoIdx : static_cast<int>(lvl);
	levelIdx = levelIdx > infoIdx ? infoIdx : levelIdx;

	static const char *levelName[] = {
		"Debug",
		"Error",
		"Warning",
		"Info"
	};

	va_list args;
	__va_start(&args, fmt);
	char fmt_[1024];
	sprintf_s(fmt_, "%s[%s]: %s %s", color, levelName[levelIdx], fmt, ANSI_COLOR_RESET);
	vfprintf(OUT_STREAM, fmt_, args);
	__crt_va_end(args);
	fprintf(OUT_STREAM, "\n");
}

int Dar::Logger::loggingLevel = static_cast<int>(LogLevel::InfoFancy);
