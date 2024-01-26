
// Collision model for RPMC operator.


#ifndef INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_HPP_
#define INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_HPP_


#include <span>
#include <cmath>      // for floor(), ceil(), round()
#include <tuple>
#include <limits>
#include <random>
#include <cstdint>    // for int64_t
#include <utility>    // for swap()
#include <variant>
#include <algorithm>  // for max()

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, type_identity<>, gsl_Expects(), gsl_Assert()

#include <makeshift/tuple.hpp>       // for value_tuple<>
#include <makeshift/functional.hpp>  // for overload<>

#include <intervals/interval.hpp>

#include <rpmc/tools/soa.hpp>            // for assertSameSoALengths()
#include <rpmc/tools/utility.hpp>        // for Span<>

#include <rpmc/operators/rpmc/common.hpp>  // for InteractionRates<>, IsSelfInteraction, DefaultLocator
#include <rpmc/operators/rpmc/models/model.hpp>  // for NoOpInteractionModel


namespace rpmc {

namespace gsl = gsl_lite;


enum class CollisionRegime : std::uint8_t
{
        // Nⱼ ≫ 1  and  Nₖ ≫ 1 , i.e. it can be assumed that any mass transfer between particle types is statistically
        // balanced through the statistical symmetry of collision rates.
    manyParticles,

        // either  Nⱼ ~ 1  or  Nₖ ~ 1 , i.e. statistical balancing of mass transfer cannot be assumed.
    fewParticles
};
constexpr auto
reflect(gsl::type_identity<CollisionRegime>)
{
    return std::array{
        CollisionRegime::manyParticles,
        CollisionRegime::fewParticles
    };
}


struct CollisionModelBaseParams
{
        // Percentual mass increase in coagulation.
        // Must be  ∈ [0, 1] .
    double relativeChangeRate = 0.05;

        // In a collision involving a tracer representing `≤ particleRegimeThreshold` particles, it cannot be assumed
        // that the symmetry of collision rates will statistically balance the mass transfer between swarms.
        // Must be  ≥ 1 .
    int particleRegimeThreshold = 100;

        // Optional separate threshold for the particle regime when computing collision rates. Experimental use only.
        // A value of  0  indicates that `particleRegimeThreshold` is used instead: the only sensible choice.
    double particleRegimeThresholdForInteractionRates = 0;
};

template <typename ParamsT = CollisionModelBaseParams>
struct CollisionModelBaseArgs : ParamsT
{
    explicit CollisionModelBaseArgs(ParamsT const& params)
        : ParamsT(params)
    {
        gsl_Expects(params.relativeChangeRate >= 0 && params.relativeChangeRate <= 1);
        gsl_Expects(params.particleRegimeThreshold >= 1);
        gsl_Expects(params.particleRegimeThresholdForInteractionRates == 0 || params.particleRegimeThresholdForInteractionRates >= 1);
    }
};

template <typename ParamsT = CollisionModelBaseParams>
struct CollisionModelArgs : CollisionModelBaseArgs<ParamsT>
{
    using CollisionModelBaseArgs<ParamsT>::CollisionModelBaseArgs;
};

template <template <typename> class TT = std::type_identity_t>
struct CollisionModelBaseParticleState
{
    TT<double> M;                    // total swarm mass                             (g)
    TT<double> m;                    // tracer mass                                  (g)
    TT<double> N;                    // number of particles in swarm

    friend constexpr auto
    reflect(gsl::type_identity<CollisionModelBaseParticleState>)
    {
        return makeshift::make_value_tuple(
            &CollisionModelBaseParticleState::M,
            &CollisionModelBaseParticleState::m,
            &CollisionModelBaseParticleState::N
        );
    }
};
using CollisionModelBaseState = CollisionModelBaseParticleState<Span>;

template <template <typename> class TT = std::type_identity_t>
struct CollisionModelBaseParticleProperties
{
    TT<double> m;       //                                       individual particle masses      (mass)
    TT<double> M;       //                                       total masses of swarms          (mass)
    TT<double> N;       // = M/m  if  m ≠ 0 , or  0  otherwise   number of particles in swarm

    friend constexpr auto
    reflect(gsl::type_identity<CollisionModelBaseParticleProperties>)
    {
        return makeshift::make_value_tuple(
            &CollisionModelBaseParticleProperties::m,
            &CollisionModelBaseParticleProperties::M,
            &CollisionModelBaseParticleProperties::N
        );
    }
};

template <typename ParamsT, template <typename> class TT>
CollisionModelBaseParticleProperties<TT>
getParticleProperties(
    CollisionModelBaseArgs<ParamsT> const& /*args*/,
    CollisionModelBaseParticleState<TT> const& ps)
{
    using namespace intervals::logic;

    return {
        .m = ps.m,
        .M = ps.M,
        .N = ps.N
    };
}


template <typename DerivedT, typename ParamsT = CollisionModelBaseParams, typename StateT = CollisionModelBaseState, typename LocatorT = DefaultLocator>
class CollisionModelBase : public NoOpInteractionModel
{
protected:
    CollisionModelArgs<ParamsT> args_;
    StateT state_;

    CollisionModelBase(ParamsT const& _params, StateT const& _state)
        : args_(_params), state_(_state)
    {
        detail::assertSameSoALengths(state_);

        if (args_.particleRegimeThresholdForInteractionRates == 0)
        {
            args_.particleRegimeThresholdForInteractionRates = args_.particleRegimeThreshold;
        }
    }

public:
    CollisionModelArgs<ParamsT> const&
    getArgs() const
    {
        return args_;
    }
    StateT const&
    getState() const
    {
        return state_;
    }

private:
        //
        // Returns the damping factor  Nₖ/βⱼₖ  which appears in the cumulative tracer–swarm collision rate
        //
        //     λᵗ⁻ˢⱼₖ = Nⱼₖ/βⱼₖ λⱼₖ = (Nₖ/βⱼₖ) (Nⱼₖ/Nₖ) λⱼₖ .
        //
    template <template <typename> class TT>
    TT<double>
    computeCollisionRateDampingFactor(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2,  // tracer, swarm
        TT<double> ffrag) const
    {
        using namespace intervals::logic;
        using namespace intervals::math;

        double qm = args_.relativeChangeRate;
        double Nth = args_.particleRegimeThresholdForInteractionRates;

        //auto dm1 = ffrag*p1.m;
        //auto dm2 = p2.m;
        //auto dmMax = max(dm1, dm2);
        //auto qmm1_dmMax = qm*p1.m/dmMax;
        auto qmm1_dmMax = min(qm*p1.m/p2.m, qm/ffrag);

            // Change  mⱼ  by no more than  qₘ mⱼ .
            //
            // β⁰ⱼₖ = max{ 1, qₘ mⱼ/δmMax } .
            //
        auto beta0 = max(1., qmm1_dmMax);

        auto result = TT<double>{ };

            // Ad-hoc modification: for swarms whose swarm particle count approaches  Nth , allow reaching
            //
            //     NthEff ≡ max{ 1, Nth-1 } .
            //
            // This prevents an extreme slowdown for buckets which include swarms with  N = Nth + ε .
        auto NthEff = std::max(1., Nth - 1.);
        //auto NthEff = Nth;

            // We have to deal with the case where  M = (Nth + ε) m  and thus  M > Nth m , but  N = M/m = Nth  due to floating-point
            // rounding. Therefore, take the boundary value  N = Nth  as a few-particles criterion only if the swarm is not also
            // classified as a many-particles swarm.
        auto manyParticlesRegime = (p1.N >= Nth) & (p2.N >= Nth);
        auto fewParticlesRegime = (p1.N <= Nth) | (p2.N <= Nth);
        if (possibly(manyParticlesRegime))
        {
                // In the many-particles regime, limit the remaining number of particles in swarm  j  such that it does not
                // cross the boundary to the few-particles regime.
                //
                //      βⱼₖ = max{  1, min{   β⁰ⱼₖ, mⱼ/mₖ (Nⱼ/Nₜₕ - 1) } }
                // ⇒ Nₖ/βⱼₖ = min{ Nₖ, max{ Nₖ/β⁰ⱼₖ, mₖ/mⱼ Nₖ/(Nⱼ/Nₜₕ - 1) } }
                //
            auto N1 = constrain(p1.N, manyParticlesRegime);
            auto N2 = constrain(p2.N, manyParticlesRegime);
            auto N2_beta = min(N2, max(N2/beta0, p2.m/p1.m*(N2/(N1/NthEff - 1))));
            assign_partial(result, N2_beta);
        }
        if (possibly_not(manyParticlesRegime) || always(fewParticlesRegime))  // few-particles regime
        {
            auto N2 = constrain(p2.N, !manyParticlesRegime);
            auto few2 = N2 <= Nth;
            if (possibly(few2))
            {
                    // No boosting if  Nₖ ≤ Nₜₕ .
                auto N2c = constrain(N2, few2);
                assign_partial(result, N2c);
            }
            if (possibly_not(few2))
            {
                    // We are in the few-particles regime, and swarm 2 is a many-particles swarm, so we can constrain
                    // swarm 1 to being a few-particles swarm.
                auto N1c = constrain(p1.N, p1.N <= Nth);

                auto N2c = constrain(N2, !few2);
                auto canDeplete = beta0*N1c >= N2c;
                auto N2c_beta0 = N2c/beta0;
                auto canDeplete_alt = N1c >= N2c_beta0;
                if (possibly(canDeplete))
                {
                        // If swarm  k  is a many-particles swarm, allow for depletion of the swarm, but only if the initial
                        // boost factor extends far enough.
                        //
                        //      βⱼₖ = Nₖ/Nⱼ
                        // ⇒ Nₖ/βⱼₖ = Nⱼ
                        //
                    auto N1eff = constrain(N1c, canDeplete_alt);
                    assign_partial(result, N1eff);
                }
                if (possibly_not(canDeplete))
                {
                        // Otherwise, constrain the boost factor such that swarm  k  does not transcend  Nth .
                        //
                        //      βⱼₖ = max{  1, min{   βⱼₖ⁰, (Nₖ - Nth)/Nⱼ } }
                        // ⇒ Nₖ/βⱼₖ = min{ Nₖ, max{ Nₖ/βⱼₖ⁰, Nⱼ/(1 - Nth/Nₖ) } }
                        //
                    auto N2eff = constrain(N2c, !canDeplete);
                    auto N2eff_beta0 = constrain(N2c_beta0, !canDeplete_alt);
                    auto N2_beta = min(N2eff, max(N2eff_beta0, N1c/(1 - NthEff/N2eff)));
                    assign_partial(result, N2_beta);
                }
            }
        }
        return result;
    }

        //
        // Returns the particle number correction factor  Nⱼₖ/Nₖ  which appears in the cumulative tracer–swarm collision rate
        //
        //     λᵗ⁻ˢⱼₖ = Nⱼₖ/βⱼₖ λⱼₖ = (Nₖ/βⱼₖ) (Nⱼₖ/Nₖ) λⱼₖ
        //
        // where  Nₖ - 1 ≤ Nⱼₖ ≤ Nₖ .
        //
    template <template <typename> class TT>
    TT<double>
    computeCollisionRateParticleNumberCorrectionFactor(
        CollisionModelBaseParticleProperties<TT> const& p1, CollisionModelBaseParticleProperties<TT> const& p2,  // tracer, swarm
        gsl::type_identity_t<TT<IsSelfInteraction>> isSelfInteraction) const
    {
        using namespace intervals::logic;
        using namespace intervals::math;

        auto result = TT<double>{ };
        auto noSwarmEmpty = (p1.N > 0.9) & (p2.N > 0.9);
        if (possibly_not(noSwarmEmpty))
        {
                //  Nⱼ  should be either  0  (for annihilated swarms or unallocated particles) or  ≥ 1 ; we consider any value
                // below  0.9  as  0 . If either particle has multiplicity  0 , no collision is possible, and collision rates
                // should be  0 .
            assign_partial(result, 0.);
        }
        if (possibly(noSwarmEmpty))
        {
            auto N1 = constrain(p1.N, noSwarmEmpty);
            auto N2 = constrain(p2.N, noSwarmEmpty);

            double Nth = args_.particleRegimeThresholdForInteractionRates;
            auto manyParticlesRegime = (N1 > Nth) & (N2 > Nth);
            if (possibly(manyParticlesRegime))
            {
                    //      Nⱼₖ = Nₖ
                    // ⇒ Nⱼₖ/Nₖ = 1
                    //
                assign_partial(result, 1.);
            }
            if (possibly_not(manyParticlesRegime))
            {
                auto N2eff = max(1., N2);

                    // In the few-particles regime, we can no longer make the assumptions that the collision partner is not the
                    // tracer itself and that the symmetry of collision rates will statistically balance the mass transfer between
                    // swarms. Instead, collisions are assumed to happen to all swarm particles at once.
                    //
                    //      Nⱼₖ = Nₖ - (1 + δⱼₖ)/2
                    // ⇒ Nⱼₖ/Nₖ = 1 - (1 + δⱼₖ)/(2 Nₖ)
                    //
                auto delta = if_else(isSelfInteraction == IsSelfInteraction::yes, 1., 0.);
                auto N_jk_N_k = 1. - (1 + delta)/(2*N2eff);
                assign_partial(result, N_jk_N_k);
            }
        }
        return result;
    }


public:
    using Locator = LocatorT;

    void
    initialize()
    {
    }

    template <template <typename> class TT = std::type_identity_t>
    struct TracerSwarmInteractionData : std::conditional_t<!std::is_same_v<LocatorT, DefaultLocator>, LocalInteractionRates<TT>, InteractionRates<TT>>
    {
        TT<double> beta_jk;     // boost factor
        TT<double> beta_kj;     // boost factor
        TT<double> ffrag = 0.;  // expected mass fraction of fragmented material in collision outcome
    };

    template <template <typename> class TT = std::type_identity_t>
    struct NonlocalParticleParticleInteractionRate
    {
        TT<double> collisionRate;
        TT<double> ffrag = 0.;          // expected mass fraction of fragmented material in collision outcome
    };
    template <template <typename> class TT = std::type_identity_t>
    struct LocalParticleParticleInteractionRate
    {
        TT<double> collisionRate;
        TT<double> ffrag = 0.;          // expected mass fraction of fragmented material in collision outcome
        TT<double> interactionWidth;
    };
    template <template <typename> class TT = std::type_identity_t>
    using ParticleParticleInteractionRate = std::conditional_t<!std::is_same_v<LocatorT, DefaultLocator>, LocalParticleParticleInteractionRate<TT>, NonlocalParticleParticleInteractionRate<TT>>;

        //
        // Returns the cumulative tracer–swarm collision rates
        //
        //     λᵗ⁻ˢⱼₖ = Nⱼₖ/βⱼₖ λⱼₖ
        //     λᵗ⁻ˢₖⱼ = Nₖⱼ/βₖⱼ λₖⱼ
        //
        // where λₖⱼ = λⱼₖ .
        //
    template <template <typename> class TT, typename ParticlePropertiesT>
    TracerSwarmInteractionData<TT>
    computeTracerSwarmInteractionData(
        ParticlePropertiesT const& p1, ParticlePropertiesT const& p2,
        TT<IsSelfInteraction> isSelfInteraction) const
    {
        using namespace intervals::logic;
        using namespace intervals::math;

        auto& self = *static_cast<DerivedT const*>(this);

        ParticleParticleInteractionRate<TT> ppRate = self.computeParticleParticleInteractionRate(p1, p2);
        auto dampingFactor_jk = computeCollisionRateDampingFactor<TT>(p1, p2, ppRate.ffrag);
        auto dampingFactor_kj = computeCollisionRateDampingFactor<TT>(p2, p1, ppRate.ffrag);
        auto particleNumberCorrectionFactor_jk = computeCollisionRateParticleNumberCorrectionFactor<TT>(p1, p2, isSelfInteraction);
        auto particleNumberCorrectionFactor_kj = computeCollisionRateParticleNumberCorrectionFactor<TT>(p2, p1, isSelfInteraction);
        gsl_Assert(always(ppRate.collisionRate >= 0));
        TracerSwarmInteractionData<TT> result;
        auto effectiveBeta_jk = dampingFactor_jk*particleNumberCorrectionFactor_jk;
        auto effectiveBeta_kj = dampingFactor_kj*particleNumberCorrectionFactor_kj;
        reset(result.interactionRate_jk, effectiveBeta_jk*ppRate.collisionRate);
        reset(result.interactionRate_kj, effectiveBeta_kj*ppRate.collisionRate);
        if constexpr (!std::is_same_v<LocatorT, DefaultLocator>)
        {
            reset(result.interactionWidth, ppRate.interactionWidth);
        }
        reset(result.beta_jk, p2.N/dampingFactor_jk);
        reset(result.beta_kj, p1.N/dampingFactor_kj);
        reset(result.ffrag, ppRate.ffrag);
        return result;
    }

    template <typename RNG, typename ParticlePropertiesT>
    void
    coagulate(
        RNG& /*randomNumberGenerator*/,
        gsl::index /*j*/, gsl::index /*k*/,
        ParticlePropertiesT const& /*p1*/, ParticlePropertiesT const& /*p2*/,
        double /*ffrag*/, double /*beta_jk*/, double /*mc*/)
    {
    }
    template <typename RNG, typename ParticlePropertiesT>
    double  // returns the mass of the sampled fragment
    fragment(
        RNG& /*randomNumberGenerator*/,
        gsl::index /*j*/, gsl::index /*k*/,
        ParticlePropertiesT const& /*p1*/, ParticlePropertiesT const& /*p2*/,
        double /*ffrag*/, double /*beta_jk*/)
    {
        gsl_FailFast();  // this function needs to be overridden by subclasses if fragmentation is to be used
    }
    template <typename RNG, typename ParticlePropertiesT>
    double  // returns the mass of the sampled fragment
    coagulateAndFragment(
        RNG& randomNumberGenerator,
        gsl::index j, gsl::index k,
        ParticlePropertiesT const& p1, ParticlePropertiesT const& p2,
        double ffrag, double beta_jk, double mc)
    {
        auto& self = *static_cast<DerivedT*>(this);

        self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
        double beta_kj = 1.;  // this is necessarily the case
        double mfrag = self.fragment(randomNumberGenerator, k, j, p2, p1, ffrag, beta_kj);
        return mfrag;
    }

    template <typename CallbackT>
    void
    transitionToFewParticlesSwarm(
        CallbackT&& callback,
        gsl::index j)
    {
        double Nth = args_.particleRegimeThreshold;

        if (state_.N[j] > 0 && state_.N[j] <= Nth)
        {
            double m1 = state_.m[j];
            while (state_.N[j] >= 1.5)
            {
                gsl::index iNew = callback.tryClone(j);
                if (iNew == -1)
                {
                        // We cannot continue if the maximal number of representative particles was exceeded.
                    gsl_Assert(false);

                        // But if we have to continue, at least stop looping.
                    break;
                }
                state_.M[iNew] = m1;
                state_.N[iNew] = 1.;
                callback.invalidate(iNew);
                state_.N[j] -= 1.;
                state_.M[j] -= m1;
            }
        }
        if (state_.N[j] == 0)
        {
                // Swarm is empty. Set particle mass to 0.
            state_.M[j] = 0;
            state_.m[j] = 0;
        }
        else
        {
                // Force  Nⱼ = 1  for the last remaining mass in the swarm.
            state_.m[j] = state_.M[j];
            state_.N[j] = 1;
        }
    }
    template <typename CallbackT, typename RNG, typename ParticlePropertiesT>
    void
    interact(
        CallbackT&& callback,
        RNG& randomNumberGenerator,
        gsl::index j, gsl::index k,
        ParticlePropertiesT const& p10, ParticlePropertiesT const& p20,
        TracerSwarmInteractionData<> const& interactionData)
    {
        auto& self = *static_cast<DerivedT*>(this);

        double Nth = args_.particleRegimeThreshold;
        double ffrag = interactionData.ffrag;

        auto collisionRegime = (p10.N <= Nth) || (p20.N <= Nth)
            ? CollisionRegime::fewParticles
            : CollisionRegime::manyParticles;
        bool swapJK = collisionRegime == CollisionRegime::fewParticles && p10.N > p20.N;
        auto& p1 = swapJK ? p20 : p10;
        auto& p2 = swapJK ? p10 : p20;
        auto beta_jk = swapJK ? interactionData.beta_kj : interactionData.beta_jk;
        auto beta_kj = swapJK ? interactionData.beta_jk : interactionData.beta_kj;
        if (swapJK)
        {
            gsl_Assert(beta_kj == 1);
            std::swap(j, k);
        }

            // Total swarm masses  (g)
        double M1 = p1.M;
        double M2 = p2.M;

        if (M1 == 0. || M2 == 0.)
        {
                // Particle has been removed. Skip.
            gsl_FailFast();  // because I think this should never happen
            //return;
        }

            // Particle masses  (g)
        double m1 = p1.m;
        double m2 = p2.m;

            // Number of particles in swarms
        double N1 = p1.N;
        double N2 = p2.N;

            // Total mass in collision, fragmented mass, cohesive mass.
        double mtot = m1 + beta_jk*m2;
        double mfrag = (m1 + 0.5*(beta_jk + 1)*m2)*(beta_jk*ffrag);
        double mc = mtot - mfrag;

        if (N1 > Nth)  // many–many
        {
            double m1New;
            if (ffrag == 0)
            {
                m1New = mc;
                self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
            }
            else
            {
                    // The representative particle will end up in the cohesive body with a probability of  Pᴄ = mᴄ/mtot .
                auto dist = std::uniform_real_distribution<>{ };
                auto rnd = dist(randomNumberGenerator);
                auto Pc = mc/mtot;
                if (rnd < Pc)
                {
                    m1New = mc;
                    self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                }
                else
                {
                    m1New = self.fragment(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk);
                }
            }
            double N1New = M1/m1New;
            state_.m[j] = m1New;
            state_.N[j] = N1New;
            if (N1New <= Nth)
            {
                transitionToFewParticlesSwarm(callback, j);
            }

            callback.invalidate(j);
        }
        else  // few–*
        {
            double m1New = mc;
            double M1New = m1New*N1;

            if (N2 <= Nth)  // few–few
            {
                    // In the few-particles regime, fragmentation works only if particles have been split up. Because coagulation is difficult
                    // to handle as well for  N > 1 , we simply won't support anything other than this.
                gsl_Assert(N1 == 1 && N2 == 1);  // exact floating-point comparison intended
                gsl_Assert(j != k);  // self-interaction of self-representing particles is not possible

                if (ffrag == 0)
                {
                    self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                    double N2New = N2 - N1;
                    gsl_Assert(N2New == 0);
                    double M2New = m2*N2New;
                    state_.N[k] = N2New;
                    state_.M[k] = M2New;
                }
                else
                {
                        // Representative particle  j  will be assumed to represent the cohesive body, and representative particle  k  will be sampled
                        // from the distribution of fragments.
                    double m2New = self.coagulateAndFragment(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                    double M2New = std::max(0., M1 - M1New + M2);  // just to guard against numerical overflow
                    double N2New = M2New/m2New;
                    state_.m[k] = m2New;
                    state_.M[k] = M2New;
                    state_.N[k] = N2New;
                    if (N2New <= Nth)
                    {
                        transitionToFewParticlesSwarm(callback, k);
                    }
                }
            }
            else  // few–many
            {
                    // The mass of the fragments is transferred to the many-particles swarm, and the representative particle of that swarm is then
                    // re-sampled from the total swarm mass.
                double M2Res = M2 - N1*beta_jk*m2;

                    // Take special care if the swarm particle count falls below 1. In this case, carry all of swarm 2 into the interaction.
                if (M2Res <= 0.9*m2)
                {
                    beta_jk = N2;
                    mtot = m1 + M2;  // assuming N1 == 1
                    mfrag = (m1 + 0.5*(beta_jk + 1)*m2)*(beta_jk*ffrag);
                    mc = mtot - mfrag;
                    m1New = mc;
                    M1New = m1New*N1;
                    M2Res = 0;
                }

                double m2New;
                double M2New = std::max(0., M1 - M1New + M2);  // just to guard against numerical overflow
                if (ffrag == 0)
                {
                    self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                    m2New = m2;
                }
                else
                {
                        // The representative particle will end up in a swarm particle with the original properties with a probability of  Pᴏ = M2Res/(M2Res + mfrag) .
                    auto dist = std::uniform_real_distribution<>{ };
                    auto rnd = dist(randomNumberGenerator);
                    auto Po = M2Res/(M2Res + mfrag);
                    if (rnd < Po)
                    {
                        self.coagulate(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                        m2New = m2;
                    }
                    else
                    {
                        m2New = self.coagulateAndFragment(randomNumberGenerator, j, k, p1, p2, ffrag, beta_jk, mc);
                        state_.m[k] = m2New;
                    }
                }
                double N2New = M2New/m2New;
                state_.M[k] = M2New;
                state_.N[k] = N2New;
                if (N2New <= Nth)
                {
                    transitionToFewParticlesSwarm(callback, k);
                }
            }
            state_.m[j] = m1New;
            state_.M[j] = M1New;
            if (m1New == 0)
            {
                    // If the entire mass ends up being fragmented, the fragments are sampled by particle  k , and particle  j  can be removed.
                state_.N[j] = 0;
            }

            callback.invalidate(j);
            callback.invalidate(k);
        }
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_OPERATORS_RPMC_MODELS_COLLISION_HPP_
