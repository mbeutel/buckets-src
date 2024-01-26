
#include <array>
#include <tuple>
#include <string>
#include <random>       // for minstd_rand
#include <utility>      // for move()
#include <string_view>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, ssize(), type_identity<>

#include <fmt/core.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <makeshift/tuple.hpp>     // for value_tuple<>
#include <makeshift/string.hpp>    // for parse_enum<>(), parse_flags<>()
#include <makeshift/variant.hpp>   // for visit(), variant_transform(), expand_failfast()

#include <rpmc/tools/particles.hpp>

#include <rpmc/simulations/event-driven.hpp>
#include <rpmc/operators/operator.hpp>
#include <rpmc/operators/rpmc-traditional.hpp>
#include <rpmc/operators/rpmc/classifiers/mass.hpp>
#include <rpmc/operators/rpmc/models/stirring.hpp>
#include <rpmc/operators/rpmc/models/friction.hpp>
#include <rpmc/operators/rpmc/models/collision/test.hpp>
#include <rpmc/operators/rpmc/models/collision/geometric.hpp>

#include "pybind11ext.hpp"
#include "py-utility.hpp"
#include "py-simulation.hpp"


void
registerBindings_RPMCTraditionalSimulation(pybind11::module m)
{
    namespace gsl = gsl_lite;
    namespace py = pybind11;
    using namespace py_rpmc;
    using namespace std::literals;


    auto pyRPMCTraditionalSimulation = pybind11ext::anonymous_class(m, "RPMCTraditionalSimulation",
        R"doc()doc",
        py::init(
            [](py::object args, py::object state, bool /*log*/, bool /*display*/)
            {
                using RNG = std::minstd_rand;

                auto effects = makeshift::parse_flags<Effects>(castAttr<std::string>({ args, "args" }, { "simulation", "effects" }));

                auto M = dataframeColumnSpan<double>({ state, "state" }, "M");
                auto m = dataframeColumnSpan<double>({ state, "state" }, "m");
                auto N = dataframeColumnSpan<double>({ state, "state" }, "N");
                auto a = dataframeColumnSpan<double>({ state, "state" }, "a");
                auto e = dataframeColumnSpan<double>({ state, "state" }, "e");
                auto sininc = dataframeColumnSpan<double>({ state, "state" }, "sininc");
                auto rho = dataframeColumnSpan<double>({ state, "state" }, "ρ");
                auto St = dataframeColumnSpan<double>({ state, "state" }, "St");
                auto vr = dataframeColumnSpan<double>({ state, "state" }, "vr");
                auto vphi = dataframeColumnSpan<double>({ state, "state" }, "vφ");
                auto hd = dataframeColumnSpan<double>({ state, "state" }, "hd");

                auto particleData = rpmc::PParticleData(rpmc::SpanParticleData(
                    std::tuple{ }, // properties that appertain to particle identity
                    std::tuple{ M, m, N, a, e, sininc, rho, St, vr, vphi, hd }));  // all other properties

                auto makeCollisionOperator = [&]() -> rpmc::PDiscreteOperator
                {
                    auto collisionKernelStr = castAttr<std::string>({ args, "args" }, { "collisions", "kernel" });
                    auto collisionKernel = makeshift::parse_enum<CollisionKernel>(collisionKernelStr);
                    auto collisionKernelV = makeshift::expand_failfast(collisionKernel);
                    auto collisionModelV = makeshift::variant_transform(
                        [&](auto collisionKernelC)
                        {
                            [[maybe_unused]] auto getCollisionModelBaseParams = [&]
                            {
                                return rpmc::CollisionModelBaseParams{
                                    .relativeChangeRate = castAttr<double>({ args, "args" }, { "simulation", "mass_growth_factor" }),
                                    .particleRegimeThreshold = castAttr<int>({ args, "args" }, { "simulation", "particle_regime_threshold" }),
                                    .particleRegimeThresholdForInteractionRates = castAttr<double>({ args, "args" }, { "simulation", "particle_regime_threshold_for_interaction_rates" })
                                };
                            };
                            [[maybe_unused]] auto getCollisionModelBaseState = [&]
                            {
                                return rpmc::CollisionModelBaseState{
                                    .M = M,
                                    .m = m,
                                    .N = N
                                };
                            };

                            if constexpr (collisionKernelC == CollisionKernel::constant)
                            {
                                return rpmc::ConstantKernelModel{
                                    rpmc::ConstantKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.collisionRate =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_collision_rate" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::linear)
                            {
                                return rpmc::LinearKernelModel{
                                    rpmc::LinearKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_collision_rate_coefficient" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::product)
                            {
                                return rpmc::ProductKernelModel{
                                    rpmc::ProductKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "product_collision_rate_coefficient" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::constantThreshold)
                            {
                                return rpmc::ConstantThresholdKernelModel{
                                    rpmc::ConstantThresholdKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.mThreshold =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_threshold_mass" }),
                                        /*.collisionRate =*/ castAttr<double>({ args, "args" }, { "collisions", "constant_collision_rate" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::linearThreshold)
                            {
                                return rpmc::LinearThresholdKernelModel{
                                    rpmc::LinearThresholdKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.mThreshold =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_threshold_mass" }),
                                        /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "linear_collision_rate_coefficient" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::runaway)
                            {
                                return rpmc::RunawayKernelModel{
                                    rpmc::RunawayKernelModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.collisionRateCoefficient =*/ castAttr<double>({ args, "args" }, { "collisions", "runaway_collision_rate_coefficient" }),
                                        /*.criticalMass =*/ castAttr<double>({ args, "args" }, { "collisions", "runaway_critical_mass" })
                                    },
                                    getCollisionModelBaseState()
                                };
                            }
                            else if constexpr (collisionKernelC == CollisionKernel::geometric)
                            {
                                return rpmc::GeometricCollisionModel{
                                    rpmc::GeometricCollisionModelParams{
                                        getCollisionModelBaseParams(),
                                        /*.MStar =*/ castAttr<double>({ args, "args" }, { "star", "M" }),
                                        /*.drMin =*/ castAttr<double>({ args, "args" }, { "zones", "ΔrMin" }),
                                        /*.restitutionCoef =*/ castAttr<double>({ args, "args" }, { "collisions", "ε" }),
                                        /*.fragmentRadius =*/ castAttr<double>({ args, "args" }, { "collisions", "Rfrag" }),
                                        /*.collisionOutcomes =*/ makeshift::parse_flags<rpmc::CollisionOutcomes>(castAttr<std::string>({ args, "args" }, { "collisions", "outcomes" })),
                                        /*.NThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "N_threshold" }),
                                        /*.StNBodyThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "St_NBody_threshold" }),
                                        /*.mNBodyThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "m_NBody_threshold" }),
                                        /*.StDustThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "St_dust_threshold" }),
                                        /*.mDustThreshold =*/ castAttr<double>({ args, "args" }, { "simulation", "m_dust_threshold" })
                                    },
                                    rpmc::GeometricCollisionModelState{
                                        getCollisionModelBaseState(),
                                        /*.a =*/ a,
                                        /*.e =*/ e,
                                        /*.sininc =*/ sininc,
                                        /*.vr =*/ vr,
                                        /*.vphi =*/ vphi,
                                        /*.hd =*/ hd,
                                        /*.St =*/ St,
                                        /*.rho =*/ rho
                                    }
                                };
                            }
                            else
                            {
                                throw std::runtime_error(
                                    fmt::format(
                                        "collision kernel '{}' not supported in standalone RPMC simulation",
                                        collisionKernelStr));
                            }
                        },
                        collisionKernelV);
                    auto collisionOperatorClassifier = rpmc::ParticleMassRPMCBaseClassifier{
                        rpmc::ParticleMassRPMCClassifierState{
                            .M = M,
                            .m = m
                        }
                    };
                    return makeshift::visit(
                        [&](auto&& collisionModel) -> rpmc::PDiscreteOperator
                        {
                            return rpmc::RPMCTraditionalOperator(
                                rpmc::RPMCTraditionalOperatorArgs{
                                },
                                particleData,
                                RNG(castAttr<unsigned int>({ args, "args" }, { "simulation", "random_seed" })),
                                std::move(collisionOperatorClassifier),
                                std::move(collisionModel));
                        },
                        collisionModelV);
                };
                auto collisionOperator = (effects & Effects::collisions) != Effects::none
                    ? makeCollisionOperator()
                    : rpmc::PDiscreteOperator(rpmc::NoOpDiscreteOperator{ });

                auto makeStirringOperator = [&]() -> rpmc::PDiscreteOperator
                {
                    auto stirringOperatorClassifier = rpmc::ParticleMassRPMCBaseClassifier{
                        rpmc::ParticleMassRPMCClassifierState{
                            .M = M,
                            .m = m
                        }
                    };
                    auto geometricModelParams = rpmc::GeometricModelParams{
                        .MStar =  castAttr<double>({ args, "args" }, { "star", "M" }),
                        .drMin = castAttr<double>({ args, "args" }, { "zones", "ΔrMin" }),
                        .relativeChangeRate = castAttr<double>({ args, "args" }, { "simulation", "velocity_growth_factor" }),
                        .maxChangeRate = castAttr<double>({ args, "args" }, { "simulation", "velocity_growth_rate" }),
                        .NThreshold = castAttr<double>({ args, "args" }, { "simulation", "N_threshold" }),
                        .StNBodyThreshold = castAttr<double>({ args, "args" }, { "simulation", "St_NBody_threshold" }),
                        .mNBodyThreshold = castAttr<double>({ args, "args" }, { "simulation", "m_NBody_threshold" }),
                        .StDustThreshold = castAttr<double>({ args, "args" }, { "simulation", "St_dust_threshold" }),
                        .mDustThreshold = castAttr<double>({ args, "args" }, { "simulation", "m_dust_threshold" }),
                        .suppress = (effects & Effects::stirring) == Effects::none
                    };
                    auto geometricModelState = rpmc::GeometricModelState{
                        .M = M,
                        .m = m,
                        .N = N,
                        .a = a,
                        .e = e,
                        .sininc = sininc,
                        .St = St,
                        .rho = rho,
                        .vr = vr,
                        .vphi = vphi,
                        .hd = hd
                    };
                    auto stirringModel = rpmc::StirringInteractionModel(geometricModelParams, geometricModelState);
                    auto frictionModel = rpmc::FrictionInteractionModel(geometricModelParams, geometricModelState);
                    return rpmc::RPMCTraditionalOperator(
                        rpmc::RPMCTraditionalOperatorArgs{
                        },
                        particleData,
                        RNG(castAttr<unsigned int>({ args, "args" }, { "simulation", "random_seed" }) + 1),
                        std::move(stirringOperatorClassifier),
                        std::move(stirringModel), std::move(frictionModel));
                };
                auto stirringOperator = (effects & (Effects::stirring | Effects::friction)) != Effects::none
                    ? makeStirringOperator()
                    : rpmc::PDiscreteOperator(rpmc::NoOpDiscreteOperator{ });

                return pybind11ext::make_unique_type<"RPMCTraditionalSimulation">(
                    rpmc::EventDrivenSimulation(
                        std::tuple{ },
                        std::tuple{ std::move(collisionOperator), std::move(stirringOperator) }));
            }),
        py::keep_alive<1, 3>{ },
        py::arg("args"), py::arg("state"), py::arg("log") = false, py::arg("display") = false);
    using CPyRPMCTraditionalSimulation = decltype(pyRPMCTraditionalSimulation)::type;
    pyRPMCTraditionalSimulation.def("run_to",
        [](CPyRPMCTraditionalSimulation& self,
           double tEnd,
           py::array_t<double> snapshotTimes, py::object snapshotCallback)
        {
            auto snapshotTimesSpan = py_rpmc::asSpan(py_rpmc::NamedObject{ snapshotTimes, "snapshot_times" });
            auto _ = py::gil_scoped_release{ };
            for (gsl::index i = 0; i < gsl::ssize(snapshotTimesSpan); ++i)
            {
                double t = snapshotTimesSpan[i];
                self.runTo(t);
                auto __ = py::gil_scoped_acquire{ };
                snapshotCallback(i, t);
            }
            self.runTo(tEnd);
        },
        py::arg("t_end"), py::arg("snapshot_times"), py::arg("snapshot_callback"));
    pyRPMCTraditionalSimulation.def("inspect",
        [](CPyRPMCTraditionalSimulation const& self,
           py::object dst,
           std::string_view quantity,
           py::object params)
        {
            auto inspectionData = rpmc::InspectionData{ };
            if (py::array_t<double>::check_(dst))
            {
                inspectionData.fdst = py_rpmc::asRaveledSpan<double, py::array::c_style>({ dst, "dst" });
            }
            else if (py::array_t<gsl::index>::check_(dst))
            {
                inspectionData.idst = py_rpmc::asRaveledSpan<gsl::index, py::array::c_style>({ dst, "dst" });
            }
            if (py::array_t<double>::check_(params))
            {
                inspectionData.fparams = py_rpmc::asRaveledSpan<double const, py::array::c_style>({ params, "params" });
            }
            else if (py::array_t<gsl::index>::check_(params))
            {
                inspectionData.iparams = py_rpmc::asRaveledSpan<gsl::index const, py::array::c_style>({ params, "params" });
            }
            bool handled = self.inspect(quantity, inspectionData);
            if (!handled)
            {
                throw std::runtime_error(fmt::format("error in inspect() call: unknown quantity '{}'", quantity));
            }
        },
        py::arg("dst"), py::arg("quantity"), py::arg("params"));
}
