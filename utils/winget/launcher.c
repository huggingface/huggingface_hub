/* Launch the bundled Python, resolving WinGet's hf.exe symlink first. */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

static BOOL WINAPI ignore_ctrl_c(DWORD event) {
    /* Python receives console events too. Let it finish before this launcher exits. */
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}

int wmain(void) {
    wchar_t path[32768];
    DWORD length = GetModuleFileNameW(NULL, path, ARRAYSIZE(path));
    if (!length || length >= ARRAYSIZE(path)) return 1;
    HANDLE file = CreateFileW(path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                              NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return 1;
    length = GetFinalPathNameByHandleW(file, path, ARRAYSIZE(path), FILE_NAME_NORMALIZED);
    CloseHandle(file);
    if (!length || length >= ARRAYSIZE(path)) return 1;
    wchar_t *separator = wcsrchr(path, L'\\');
    if (!separator) return 1;
    *separator = L'\0';
    /* Drop the \\?\ prefix: Python would keep it in sys.executable and pass it to subprocesses. */
    wchar_t *directory = path;
    if (wcsncmp(path, L"\\\\?\\UNC\\", 8) == 0) {
        directory = path + 6;
        directory[0] = L'\\';
    } else if (wcsncmp(path, L"\\\\?\\", 4) == 0) {
        directory = path + 4;
    }

    /* Run main() with sys.argv[0] = 'hf' so usage and error messages show `hf`, not `python -m ...`.
       Preserve the original argument quoting, including spaces and trailing backslashes. */
    const wchar_t *args = GetCommandLineW();
    if (*args == L'"') {
        ++args;
        while (*args && *args != L'"') ++args;
        if (*args) ++args;
    } else {
        while (*args && *args != L' ' && *args != L'\t') ++args;
    }
    wchar_t command[32768];
    if (_snwprintf_s(command, ARRAYSIZE(command), _TRUNCATE,
                    L"\"%ls\\python\\python.exe\" -I -c \"import sys; sys.argv[0] = 'hf'; "
                    L"from huggingface_hub.cli.hf import main; main()\" %ls",
                    directory, args) < 0) return 1;
    STARTUPINFOW startup = {0};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process = {0};
    SetConsoleCtrlHandler(ignore_ctrl_c, TRUE);
    if (!CreateProcessW(NULL, command, NULL, NULL, TRUE, 0, NULL, NULL, &startup, &process)) {
        fwprintf(stderr, L"Cannot start the bundled hf Python runtime (Windows error %lu).\n", GetLastError());
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hProcess);
    return (int)exit_code;
}
