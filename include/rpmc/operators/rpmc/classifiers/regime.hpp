
#ifndef INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_REGIME_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_REGIME_HPP_


#include <cstdint>  // for int8_t

#include <gsl-lite/gsl-lite.hpp>  // for gsl_DEFINE_ENUM_RELATIONAL_OPERATORS()


namespace rpmc {


enum ParticleRegimeClassification : std::int8_t
{
    manyParticles,
    fewParticles
};
gsl_DEFINE_ENUM_RELATIONAL_OPERATORS(ParticleRegimeClassification)


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_CLASSIFIERS_REGIME_HPP_
