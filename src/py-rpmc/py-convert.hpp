
#ifndef INCLUDED_RPMC_PY_CONVERT_HPP_
#define INCLUDED_RPMC_PY_CONVERT_HPP_


#include <pybind11/pybind11.h>

#include <rpmc/operators/rpmc/buckets/log.hpp>


namespace py_rpmc {

namespace py = pybind11;


rpmc::LogBucketingParams
toLogBucketingParams(NamedObject<> obj);


} // namespace py_rpmc


#endif // INCLUDED_RPMC_PY_CONVERT_HPP_
