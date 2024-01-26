
#ifndef INCLUDED_PYBIND11EXT_HPP_
#define INCLUDED_PYBIND11EXT_HPP_


#include <string>
#include <memory>       // for unique_ptr<>, shared_ptr<>
#include <utility>      // for move(), forward<>()
#include <type_traits>  // for remove_cv<>, remove_reference<>

#include <pybind11/pybind11.h>


namespace pybind11ext {

namespace detail {


template <typename Signature> struct function_return_type;
template <typename Return, typename... Args> struct function_return_type<Return(Args...)> { using type = Return; };
template <typename Return, typename... Args> struct function_return_type<Return (*)(Args...)> { using type = Return; };
template <typename Signature> using function_return_type_t = typename function_return_type<Signature>::type;

template <typename T, typename... ExtraT> struct class__ { using type = pybind11::class_<T, ExtraT...>; };
template <typename T, typename... ExtraT> struct class__<std::unique_ptr<T>, ExtraT...> { using type = pybind11::class_<T, /*std::unique_ptr<T>,*/ ExtraT...>; };
template <typename T, typename... ExtraT> struct class__<std::shared_ptr<T>, ExtraT...> { using type = pybind11::class_<T, std::shared_ptr<T>, ExtraT...>; };
template <typename T, typename... ExtraT> using class_ = typename class__<T, ExtraT...>::type;


template <typename V>
class opaque
{
private:
    V data_;

public:
    using type = V;

    constexpr explicit opaque(V&& _data)
        : data_(std::move(_data))
    {
    }

    constexpr V&
    operator ()(void) & { return data_; }
    constexpr V const&
    operator ()(void) const & { return data_; }
    constexpr V&&
    operator ()(void) && { return std::move(data_); }
};

template <std::size_t N>
struct fixed_string
{
    char buf[N + 1] = { };

    constexpr fixed_string(char const* s)
    {
        for (std::size_t i = 0; i != N; ++i)
        {
            buf[i] = s[i];
        }
    }
    constexpr operator char const*() const
    {
        return buf;
    }

    friend auto
    operator<=>(fixed_string const& lhs, fixed_string const& rhs) = default;
};
template <std::size_t N> fixed_string(char const (&)[N]) -> fixed_string<N - 1>;


template <fixed_string Label, typename T>
class unique_type : public T
{
public:
    using type = T;

    constexpr explicit unique_type(T&& _data)
        : T(std::move(_data))
    {
    }
};


} // namespace detail


template <typename T>
[[nodiscard]] detail::opaque<std::remove_reference_t<std::remove_cv_t<T>>>
make_opaque(T&& arg)
{
    return detail::opaque<std::remove_reference_t<std::remove_cv_t<T>>>(std::forward<T>(arg));
}


template <detail::fixed_string Label, typename T>
[[nodiscard]] detail::unique_type<Label, std::remove_reference_t<std::remove_cv_t<T>>>
make_unique_type(T&& arg)
{
    return detail::unique_type<Label, std::remove_reference_t<std::remove_cv_t<T>>>(std::forward<T>(arg));
}


template <typename Func, typename... FactoryArgs, typename... Extra>
detail::class_<detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>>
anonymous_class(
    pybind11::handle scope, const char *name,
    pybind11::detail::initimpl::factory<Func, FactoryArgs...> &&init, const Extra &... extra)
{
    using R = detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>;
    auto cls = detail::class_<R>(scope, name);
    cls.def(std::move(init), extra...);
    return cls;
}

template <typename Func, typename... FactoryArgs, typename... Extra>
detail::class_<detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>>
anonymous_class(
    pybind11::handle scope, const char *name, const char *doc,
    pybind11::detail::initimpl::factory<Func, FactoryArgs...> &&factory, const Extra &... extra)
{
    using R = detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>;
    auto cls = detail::class_<R>(scope, name, doc);
    cls.def(std::move(factory), extra...);
    return cls;
}

template <typename Scope, typename Func, typename... Extra>
detail::class_<detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>>
def_anonymous(
    Scope &scope, const char *name,
    Func func, const Extra &... extra)
{
    using R = detail::function_return_type_t<pybind11::detail::function_signature_t<Func>>;
    auto retName = std::string("__") + name + "_result";
    auto cls = detail::class_<R>(scope, retName.c_str());
    scope.def(name, std::move(func), extra...);
    return cls;
}


} // namespace pybind11ext


#endif // INCLUDED_PYBIND11EXT_HPP_
