@ECHO OFF
SET SCRIPTDIR=%~dp0

%SCRIPTDIR%\..\tools\resourcecompiler\resourcecompiler.exe %SCRIPTDIR%\res %SCRIPTDIR%\res shaders
