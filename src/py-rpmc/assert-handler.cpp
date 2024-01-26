
//#define ASSERT_HANDLER_MESSAGEBOX

#include <string>
#include <cstring>
#include <exception>  // for terminate()

#include <fmt/core.h>


#if defined(_WIN32) && defined(ASSERT_HANDLER_MESSAGEBOX)
# define WIN32_LEAN_AND_MEAN
# include <windows.h>
#else
# include <iostream>
#endif

#include <gsl-lite/gsl-lite.hpp>


#ifdef _WIN32
 //extern "C" int __stdcall IsDebuggerPresent(void);
# ifdef _MSC_VER
    // prefer compiler intrinsic over function call
#  include <intrin.h>
#  define DEBUG_BREAK_() __debugbreak()
# else // _MSC_VER
  extern "C" void __stdcall DebugBreak(void);
#  define DEBUG_BREAK_() DebugBreak()
# endif // _MSC_VER
# else // _WIN32
    // assume POSIX
# if defined(__clang__) || defined(__GNUC__)
    // prefer inline assembly over function call
#  if defined(__i386__) || defined(__x86_64__)
#   define DEBUG_BREAK_() __asm__("int $3")
#  else // defined(__i386__) || defined(__x86_64__)
#   include <signal.h>
#   define DEBUG_BREAK_() raise(SIGINT)
#  endif // defined(__i386__) || defined(__x86_64__)
# endif // defined(__clang__) || defined(__GNUC__)
#endif // _WIN32

    //ᅟ
    // Use this macro to insert a debug break. This can be useful to locate a particular piece of code in the debugger despite
    // optimization, e.g. for analyzing the generated machine code.
    //
#define DEBUG_BREAK() DEBUG_BREAK_()


gsl_api void
gsl::fail_fast_assert_handler(char const* expression, char const* message, char const* file, int line)
{
#if defined(_WIN32) && defined(ASSERT_HANDLER_MESSAGEBOX)
    auto msg = fmt::format(
        "{} in {}({}): {}\n\nPress abort to terminate, retry to break into the debugger, or ignore to continue execution.",
        message, file, line, expression);
    auto id = ::MessageBoxA(NULL, msg.c_str(), message, MB_ABORTRETRYIGNORE | MB_ICONERROR | MB_SETFOREGROUND);
    switch (id)
    {
    case IDABORT:
        std::terminate();
        break;

    case IDRETRY:
        DEBUG_BREAK();
        break;

    case IDIGNORE:
        break;
    }
#else  // !(defined(_WIN32) && defined(ASSERT_HANDLER_MESSAGEBOX))
    auto msg = fmt::format(
        "{} in {}({}): {}\n\nTerminate (1), break into debugger (2), or continue execution (3)?\n",
        message, file, line, expression);
    std::cerr << msg;
    std::string input;
    long long id;
    while (true)
    {
        if (!std::getline(std::cin, input))
        {
            std::cerr << "Error: unable to read from stdin.\n" << std::flush;
            std::terminate();
        }
        char* endPtr = nullptr;
        id = std::strtoll(input.c_str(), &endPtr, 10);
        if (id >= 1 && id <= 3) break;
        std::cerr << "Invalid input; expected '1', '2', or '3'.\n";
    }
    switch (id)
    {
    case 1:
        std::terminate();
        break;

    case 2:
        DEBUG_BREAK();
        break;

    case 3:
        break;
    }
#endif  // defined(_WIN32) && defined(ASSERT_HANDLER_MESSAGEBOX)
}
