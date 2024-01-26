
#ifndef INCLUDED_RPMC_TOOLS_BUCKETING_HPP_
#define INCLUDED_RPMC_TOOLS_BUCKETING_HPP_


#include <span>
#include <vector>
#include <ranges>
#include <limits>
#include <utility>      // for move(), swap()
#include <cstddef>      // for size_t
#include <cstdint>      // for int32_t
#include <numeric>      // for accumulate()
#include <concepts>
#include <optional>
#include <algorithm>    // for lower_bound()
#include <type_traits>  // for integral_constant<>, conditional<>

#include <gsl-lite/gsl-lite.hpp>  // for dim, index, narrow_cast<>(), finally(), gsl_Expects(), gsl_Assert()

#include <intervals/interval.hpp>  // for interval<>

#include <rpmc/tools/expandoarray.hpp>


namespace rpmc {

namespace gsl = gsl_lite;

namespace detail {


template <typename T, std::convertible_to<T> U>
void
push_back_if_neq_last(std::vector<T>& v, U&& x)
{
    if (v.empty() || x != v.back())
    {
        v.push_back(std::forward<U>(x));
    }
}

template <typename T> struct is_std_array : std::false_type { };
template <typename T, std::size_t N> struct is_std_array<std::array<T, N>> : std::true_type { };
template <typename T> constexpr bool is_std_array_v = false;
template <typename T, std::size_t N> constexpr bool is_std_array_v<std::array<T, N>> = true;
template <typename T> concept std_array = is_std_array_v<T>;

struct inaccessible
{
    inaccessible() = delete;
};

template <typename T, typename OtherT>
struct other_effect
{
    using type = OtherT;
};
template <typename T>
struct other_effect<T, T>
{
    using type = inaccessible;
};
template <typename T, typename OtherT>
using other_effect_t = typename other_effect<T, OtherT>::type;

template <template <typename> class EffectT, typename ContainerT>
struct effect_iterator : std::conditional<std::is_const_v<EffectT<int>>, typename ContainerT::const_iterator, typename ContainerT::iterator>
{
};
template <template <typename> class EffectT, typename ContainerT>
using effect_iterator_t = typename effect_iterator<EffectT, ContainerT>::type;


} // namespace detail


template <typename T>
concept CBucketClassifier = requires(T const& cx, gsl::index i)
{
        // Returns `BucketLabelT` or `std::optional<BucketLabelT>`.
    { cx.classify(i) };
};

template <typename T, typename BucketingSchemeT>
concept CLocalBucketClassifier = CBucketClassifier<T> && requires(T const& cx, typename BucketingSchemeT::ConstBucket cbucket, gsl::index i)
{
        // Returns the sub-bucket of the given entry in the given bucket.
    { cx.subclassify(cbucket, i) } -> std::same_as<typename BucketingSchemeT::SubBucketLabel>;
};

template <typename T, typename BucketingSchemeT>
concept CLocalBucketUpdater = requires(
    T& x, typename BucketingSchemeT::Bucket bucket,
    std::span<typename BucketingSchemeT::SubBucketUpdate const> subBucketUpdates)
{
    { x.onBucketAdded(bucket, subBucketUpdates) } -> std::same_as<void>;
    { x.onBucketUpdated(bucket, subBucketUpdates) } -> std::same_as<void>;
    { x.onRemovingBucket(bucket, subBucketUpdates) } -> std::same_as<void>;
};
template <typename T, typename BucketingSchemeT>
concept CNonlocalBucketUpdater = requires(
    T& x, typename BucketingSchemeT::Bucket bucket)
{
    { x.onBucketAdded(bucket) } -> std::same_as<void>;
    { x.onBucketUpdated(bucket) } -> std::same_as<void>;
    { x.onRemovingBucket(bucket) } -> std::same_as<void>;
};
template <typename T, typename BucketingSchemeT>
concept CBucketUpdater = requires(
    T& x, typename BucketingSchemeT::Bucket bucket, gsl::index i,
    std::span<typename BucketingSchemeT::SubBucketUpdate const> subBucketUpdates)
{
    { BucketingSchemeT::haveLocality } -> std::convertible_to<bool>;
    { x.onEntryAddedToBucket(bucket, i) } -> std::same_as<void>;
    { x.onEntryUpdatedInBucket(bucket, i) } -> std::same_as<void>;
    { x.onRemovingEntryFromBucket(bucket, i) } -> std::same_as<void>;
} && (( BucketingSchemeT::haveLocality && CLocalBucketUpdater<T, BucketingSchemeT>)
   || (!BucketingSchemeT::haveLocality && CNonlocalBucketUpdater<T, BucketingSchemeT>));



template <typename BucketingSchemeT>
class DefaultBucketUpdater
{
    using Bucket = typename BucketingSchemeT::Bucket;
    using SubBucketUpdate = typename BucketingSchemeT::SubBucketUpdate;

public:
    void
    onEntryAddedToBucket(Bucket /*bucket*/, gsl::index /*i*/)
    {
    }
    void
    onEntryUpdatedInBucket(Bucket /*bucket*/, gsl::index /*i*/)
    {
    }
    void
    onRemovingEntryFromBucket(Bucket /*bucket*/, gsl::index /*i*/)
    {
    }

    void
    onBucketAdded(Bucket /*bucket*/, std::span<SubBucketUpdate const> /*subBucketUpdates*/ = { })
    {
    }
    void
    onBucketUpdated(Bucket /*bucket*/, std::span<SubBucketUpdate const> /*subBucketUpdates*/ = { })
    {
    }
    void
    onRemovingBucket(Bucket /*bucket*/, std::span<SubBucketUpdate const> /*subBucketUpdates*/ = { })
    {
    }
};


template <typename BucketingSchemeT>
class DefaultBucketClassifier
{
    using ConstBucket = typename BucketingSchemeT::ConstBucket;
    using BucketLabel = typename BucketingSchemeT::BucketLabel;
    using SubBucketLabel = typename BucketingSchemeT::SubBucketLabel;

public:
    BucketLabel
    classify(gsl::index /*j*/) const
    {
        return { };
    }
    SubBucketLabel
    subclassify(ConstBucket /*bucket*/, gsl::index /*j*/) const
    {
        return { };
    }
};


template <typename BucketLabelT, typename BucketDataT = void, typename BucketBucketDataT = void,
          typename SubBucketLabelT = std::array<std::int32_t, 0>, typename SubBucketDataT = void>
class BucketingScheme;
template <typename BucketLabelT, typename BucketDataT, typename BucketBucketDataT, typename LocationIndexT, std::size_t LocationDim, typename SubBucketDataT>
class BucketingScheme<BucketLabelT, BucketDataT, BucketBucketDataT, std::array<LocationIndexT, LocationDim>, SubBucketDataT>
{
    static_assert(LocationDim >= 0);

public:
    struct Empty { };

    using Index = std::int32_t;
    using Dim = Index;
    using Diff = Index;
    using SubBucketLabel = std::array<LocationIndexT, LocationDim>;

    using BucketData = std::conditional_t<std::is_void_v<BucketDataT>, Empty, BucketDataT>;
    using BucketBucketData = std::conditional_t<std::is_void_v<BucketBucketDataT>, Empty, BucketBucketDataT>;
    using SubBucketData = std::conditional_t<std::is_void_v<SubBucketDataT>, Empty, SubBucketDataT>;

    static constexpr bool haveLocality = LocationDim > 0;

private:
    struct SubBucketRecord
    {
        std::vector<Index> entries;
        [[no_unique_address]] SubBucketData subBucketData;

        bool
        empty() const
        {
            return entries.empty();
        }
        gsl::dim
        ssize() const
        {
            return std::ssize(entries);
        }
    };
    using NonlocalEntries = SubBucketRecord;
    struct LocalEntries
    {
        ExpandoArray<SubBucketRecord, LocationDim, LocationIndexT> subBucketRecords;
        Dim n = 0;

        constexpr bool
        empty() const
        {
            return n == 0;
        }
        constexpr gsl::dim
        ssize() const
        {
            return n;
        }

        auto
        allEntries() const
        {
            return subBucketRecords.entries()
                 | std::views::transform(
                     [](SubBucketRecord const& subBucketRecord)
                     {
                         return std::span<Index const>(subBucketRecord.entries);
                     })
                 | std::views::join;
        }
    };
    using Entries = std::conditional_t<haveLocality, LocalEntries, NonlocalEntries>;

    struct BucketFlags
    {
        bool justAdded: 1 = false;
        bool noLocalityYet: 1 = true;
    };
    struct BucketRecord
    {
        Entries entries;
        std::vector<BucketBucketData> bucketBucketData;
        BucketLabelT label;
        BucketFlags flags;
        [[no_unique_address]] BucketData bucketData;

        friend void
        swap(BucketRecord& lhs, BucketRecord& rhs)
        {
            using std::swap;
            swap(lhs.entries, rhs.entries);
            swap(lhs.bucketBucketData, rhs.bucketBucketData);
            swap(lhs.label, rhs.label);
            swap(lhs.flags, rhs.flags);
            swap(lhs.bucketData, rhs.bucketData);
        }
    };
    struct NonlocalBucketEntryIndex
    {
    private:
        Index index_;  // a negative index is interpreted as an index  `-(index_+1)`  into the `inactiveEntries_` array

    public:
        explicit NonlocalBucketEntryIndex(std::true_type, gsl::index i, SubBucketLabel = { })
            : index_(gsl::narrow_cast<Index>(i))
        {
        }
        explicit NonlocalBucketEntryIndex(std::false_type, gsl::index i)
            : index_(gsl::narrow_cast<Index>(-(i + 1)))
        {
        }
        bool
        active() const
        {
            return index_ >= 0;
        }
        gsl::index
        index() const
        {
            return index_ >= 0
                ? index_
                : -(index_ + 1);
        }
        SubBucketLabel
        subBucketLabel() const
        {
            return { };
        }
    };
    struct LocalBucketEntryIndex
    {
    private:
        Index index_;  // a negative index is interpreted as an index  `-(index_+1)`  into the `inactiveEntries_` array
        SubBucketLabel subBucketLabel_;

    public:
        explicit LocalBucketEntryIndex(std::true_type, gsl::index i, SubBucketLabel iLoc)
            : index_(gsl::narrow_cast<Index>(i)), subBucketLabel_(iLoc)
        {
        }
        explicit LocalBucketEntryIndex(std::false_type, gsl::index i)
            : index_(gsl::narrow_cast<Index>(-(i + 1))), subBucketLabel_{ }
        {
        }
        bool
        active() const
        {
            return index_ >= 0;
        }
        gsl::index
        index() const
        {
            return index_ >= 0
                ? index_
                : -(index_ + 1);
        }
        SubBucketLabel const&
        subBucketLabel() const
        {
            return subBucketLabel_;
        }
    };
    using BucketEntryIndex = std::conditional_t<haveLocality, LocalBucketEntryIndex, NonlocalBucketEntryIndex>;
    struct EntryRecord
    {
        BucketLabelT bucketLabel;
        BucketEntryIndex bucketEntryIndex;

        bool
        active() const
        {
            return bucketEntryIndex.active();
        }
    };
    struct BucketRef
    {
        BucketLabelT label;  // TODO: we could avoid storing this at the cost of an additional indirection
        Index bucketDataIndex;
    };

    struct BucketUpdate
    {
        Index bucketIndex;
        SubBucketLabel subBucketLabel;
        Diff delta;

        bool operator ==(BucketUpdate const& rhs) const = default;
        constexpr auto operator <=>(BucketUpdate const& rhs) const
        {
            auto bucketIndexCmp = rhs.bucketIndex <=> bucketIndex;  // reverse order
            if (std::is_neq(bucketIndexCmp))
            {
                return bucketIndexCmp;
            }
            auto deltaCmp = delta <=> rhs.delta;
            if (std::is_neq(deltaCmp))
            {
                return deltaCmp;
            }
            auto subBucketLabelCmp = subBucketLabel <=> rhs.subBucketLabel;
            return subBucketLabelCmp;
        }
    };

public:
    struct SubBucketUpdate
    {
        SubBucketLabel subBucketLabel;
        Diff delta;
    };

private:
    std::vector<EntryRecord> entryRecords_;
    std::vector<BucketRef> orderedBucketRefs_;
    std::vector<BucketRecord> bucketRecords_;
    std::vector<Index> inactiveEntries_;

        // Temporary data
    bool updating_ = false;
    std::vector<BucketUpdate> bucketUpdates_;
    [[no_unique_address]] std::conditional_t<haveLocality, std::vector<SubBucketUpdate>, Empty> subBucketUpdates_;

    static std::span<SubBucketUpdate const>
    coalesceSubBucketUpdates(
        std::vector<SubBucketUpdate>& subBucketUpdates,
        typename std::vector<BucketUpdate>::const_iterator firstBucketUpdate, typename std::vector<BucketUpdate>::const_iterator lastBucketUpdate)
    {
        gsl_AssertDebug(firstBucketUpdate != lastBucketUpdate);

        subBucketUpdates.clear();
        subBucketUpdates.push_back(
            SubBucketUpdate{
                .subBucketLabel = firstBucketUpdate->subBucketLabel,
                .delta = firstBucketUpdate->delta
            });
        for (auto it = std::next(firstBucketUpdate); it != lastBucketUpdate; ++it)
        {
            if (it->subBucketLabel == subBucketUpdates.back().subBucketLabel)
            {
                subBucketUpdates.back().delta += it->delta;
            }
            else
            {
                if (subBucketUpdates.back().delta == 0)
                {
                    subBucketUpdates.pop_back();
                }
                subBucketUpdates.push_back(
                    SubBucketUpdate{
                        .subBucketLabel = it->subBucketLabel,
                        .delta = it->delta
                    });
            }
        }
        if (subBucketUpdates.back().delta == 0)
        {
            subBucketUpdates.pop_back();
        }

        return subBucketUpdates;
    }

    typename std::vector<BucketRef>::iterator
    findBucketRef(BucketLabelT const& label)
    {
        auto it = std::lower_bound(orderedBucketRefs_.begin(), orderedBucketRefs_.end(), label,
            [](BucketRef const& element, BucketLabelT const& value)
            {
                return element.label < value;
            });
        gsl_Assert(it != orderedBucketRefs_.end());
        gsl_Assert(it->label == label);
        return it;
    }
    typename std::vector<BucketRef>::const_iterator
    findBucketRef(BucketLabelT const& label) const
    {
        auto it = std::lower_bound(orderedBucketRefs_.begin(), orderedBucketRefs_.end(), label,
            [](BucketRef const& element, BucketLabelT const& value)
            {
                return element.label < value;
            });
        gsl_Assert(it != orderedBucketRefs_.end());
        gsl_Assert(it->label == label);
        return it;
    }

    static auto
    makeNewBucketEntries(gsl::index i)
    {
        if constexpr (haveLocality)
        {
            auto result = LocalEntries{ .n = 1 };
            result.subBucketRecords.assign(SubBucketLabel{ }, SubBucketRecord{ .entries = { gsl::narrow_cast<Index>(i) } });
            return result;
        }
        else
        {
            return NonlocalEntries{ .entries = { gsl::narrow_cast<Index>(i) } };
        }
    }
    SubBucketRecord&
    getSubBucketRecord(BucketRecord& bucketRecord, LocalBucketEntryIndex const& bucketEntryIndex)
    {
        return bucketRecord.entries.subBucketRecords[bucketEntryIndex.subBucketLabel()];
    }
    SubBucketRecord&
    getSubBucketRecord(BucketRecord& bucketRecord, NonlocalBucketEntryIndex const&)
    {
        return bucketRecord.entries;
    }

    template <CBucketUpdater<BucketingScheme> UpdaterT, CBucketClassifier ClassifierT>
    void
    doUpdateEntry(UpdaterT&& updater, ClassifierT&& classifier, gsl::index i)
    {
        gsl_Expects(i >= 0 && i < std::ssize(entryRecords_));

        auto& entryRecord = entryRecords_[i];
        std::optional<BucketLabelT> maybeNewBucketLabel = classifier.classify(i);
        auto oldEntryRecord = entryRecord;
        bool wasActive = oldEntryRecord.active();
        bool isActive = maybeNewBucketLabel.has_value();
        bool sameBucket = wasActive && isActive && *maybeNewBucketLabel == oldEntryRecord.bucketLabel;
        bucketUpdates_.reserve(bucketUpdates_.size() + 2);
        [[maybe_unused]] bool removeFromSubBucket = false;
        if (wasActive && !isActive)
        {
            if constexpr (haveLocality)
            {
                removeFromSubBucket = true;
            }

                // Add particle to list of inactive entries.
            entryRecord.bucketLabel = { };
            gsl::index newBucketEntryIndex = std::ssize(inactiveEntries_);
            inactiveEntries_.push_back(gsl::narrow_cast<Index>(i));
            entryRecord.bucketEntryIndex = BucketEntryIndex(std::false_type{ }, newBucketEntryIndex);
        }
        else if (isActive && !sameBucket)
        {
                // The bucket changed, so we look for the new bucket, add one if it doesn't exist yet, and add the particle
                // entry to the new bucket.
            auto& newBucketLabel = *maybeNewBucketLabel;
            auto bucketRefIt = std::lower_bound(orderedBucketRefs_.begin(), orderedBucketRefs_.end(), newBucketLabel,
                [](BucketRef const& element, BucketLabelT const& value)
                {
                    return element.label < value;
                });
            gsl::index bucketEntryIndex;
            auto subBucketLabel = SubBucketLabel{ };
            if (bucketRefIt == orderedBucketRefs_.end() || bucketRefIt->label != newBucketLabel)
            {
                    // Bucket doesn't exist yet; add it.
                Dim oldNumBuckets = gsl::narrow_cast<Dim>(std::ssize(orderedBucketRefs_));
                Dim newNumBuckets = oldNumBuckets + 1;

                    // Make sure to first allocate memory for the new bucket everywhere before actually adding it.
                for (auto& bucketRecord : bucketRecords_)
                {
                    bucketRecord.bucketBucketData.reserve(newNumBuckets);
                }
                bucketRecords_.reserve(newNumBuckets);
                bucketEntryIndex = 0;
                auto newEntries = makeNewBucketEntries(i);
                auto newBucketBucketData = std::vector<BucketBucketData>(newNumBuckets);

                    // Now add the bucket. We need a stable iterator for `insert()`, so we cannot call `reserve()` on
                    // `orderedBucketRefs_`, therefore this has to be the last memory-allocating operation in order to
                    // stay exception-safe.
                bucketRefIt = orderedBucketRefs_.insert(bucketRefIt, BucketRef{
                    .label = newBucketLabel,
                    .bucketDataIndex = oldNumBuckets
                });
                for (auto& bucketRecord : bucketRecords_)
                {
                    bucketRecord.bucketBucketData.push_back({ });
                }
                bucketRecords_.push_back(BucketRecord{
                    .entries = std::move(newEntries),
                    .bucketBucketData = std::move(newBucketBucketData),
                    .label = newBucketLabel,
                    .flags = BucketFlags{
                        .justAdded = true,
                        .noLocalityYet = true
                    }
                });
            }
            else
            {
                    // Bucket already exists. Add particle entry.
                auto bucketRecordIt = bucketRecords_.begin() + bucketRefIt->bucketDataIndex;
                auto& bucketRecord = *bucketRecordIt;
                if constexpr (haveLocality)
                {
                    if (!bucketRecord.flags.noLocalityYet)
                    {
                        subBucketLabel = classifier.subclassify(ConstBucket(bucketRecordIt), i);
                    }
                    auto& subBucketEntries = bucketRecord.entries.subBucketRecords.obtain(subBucketLabel);
                    bucketEntryIndex = std::ssize(subBucketEntries.entries);
                    subBucketEntries.entries.push_back(gsl::narrow_cast<Index>(i));
                    ++bucketRecord.entries.n;
                }
                else
                {
                    bucketEntryIndex = std::ssize(bucketRecord.entries.entries);
                    bucketRecord.entries.entries.push_back(gsl::narrow_cast<Index>(i));
                }
            }
            entryRecord.bucketLabel = newBucketLabel;
            entryRecord.bucketEntryIndex = BucketEntryIndex(std::true_type{ }, bucketEntryIndex, subBucketLabel);

                // Note down the bucket for updating.
            bucketUpdates_.push_back(
                BucketUpdate{
                    .bucketIndex = bucketRefIt->bucketDataIndex,
                    .subBucketLabel = subBucketLabel,
                    .delta = +1
                });

            updater.onEntryAddedToBucket(Bucket(bucketRecords_.begin() + bucketRefIt->bucketDataIndex), i);
        }
        else if (sameBucket)
        {
            auto bucketIt = findBucketRef(*maybeNewBucketLabel);
            auto& bucketRef = *bucketIt;

            SubBucketLabel subBucketLabel = { };
            Diff delta = 0;
            if constexpr (haveLocality)
            {
                auto bucketRecordIt = bucketRecords_.begin() + bucketRef.bucketDataIndex;
                auto& bucketRecord = *bucketRecordIt;
                if (!bucketRecord.flags.noLocalityYet)
                {
                        // If necessary, move the entry to a different sub-bucket.
                    subBucketLabel = classifier.subclassify(ConstBucket(bucketRecordIt), i);
                    if (subBucketLabel != entryRecord.bucketEntryIndex.subBucketLabel())
                    {
                        delta = +1;
                        removeFromSubBucket = true;

                        auto& subBucketEntries = bucketRecord.entries.subBucketRecords.obtain(subBucketLabel);
                        auto bucketEntryIndex = std::ssize(subBucketEntries.entries);
                        subBucketEntries.entries.push_back(gsl::narrow_cast<Index>(i));
                        ++bucketRecord.entries.n;
                        entryRecord.bucketEntryIndex = BucketEntryIndex(std::true_type{ }, bucketEntryIndex, subBucketLabel);
                    }
                }
            }

            updater.onEntryUpdatedInBucket(Bucket(bucketRecords_.begin() + bucketRef.bucketDataIndex), i);

                // Note down the bucket for updating.
            bucketUpdates_.push_back(
                BucketUpdate{
                    .bucketIndex = bucketRef.bucketDataIndex,
                    .subBucketLabel = subBucketLabel,
                    .delta = delta
                });
        }
        if (isActive && !wasActive)
        {
                // Remove from list of inactive entries.
                // If the entry isn't at the end of the list, use a "castling" maneuver to remove it in  𝒪(1)  steps.
            gsl::index oldEntryIndex = oldEntryRecord.bucketEntryIndex.index();
            if (oldEntryIndex != std::ssize(inactiveEntries_) - 1)
            {
                using std::swap;
                auto& swapEntry = inactiveEntries_[oldEntryIndex];
                swap(swapEntry, inactiveEntries_.back());
                entryRecords_[swapEntry].bucketEntryIndex = BucketEntryIndex(std::false_type{ }, oldEntryIndex);
            }
            inactiveEntries_.pop_back();
        }
        else if (wasActive && (!sameBucket || removeFromSubBucket))
        {
            Index oldBucketIndex = findBucketRef(oldEntryRecord.bucketLabel)->bucketDataIndex;
            if (!sameBucket)
            {
                updater.onRemovingEntryFromBucket(Bucket(bucketRecords_.begin() + oldBucketIndex), i);
            }

                // Remove the entry from the old bucket.
                // If the entry isn't at the end of the list, use a "castling" maneuver to remove it in  𝒪(1)  steps.
            BucketRecord& oldBucketRecord = bucketRecords_[oldBucketIndex];
            auto& oldSubBucketRecord = getSubBucketRecord(oldBucketRecord, oldEntryRecord.bucketEntryIndex);
            gsl::index oldEntryIndex = oldEntryRecord.bucketEntryIndex.index();
            if (oldEntryIndex != std::ssize(oldSubBucketRecord.entries) - 1)
            {
                using std::swap;
                auto& swapEntry = oldSubBucketRecord.entries[oldEntryIndex];
                swap(swapEntry, oldSubBucketRecord.entries.back());
                //entryRecords_[swapEntry].bucketEntryIndex = BucketEntryIndex(std::true_type{ }, oldEntryIndex);
                entryRecords_[swapEntry].bucketEntryIndex = oldEntryRecord.bucketEntryIndex;
            }
            oldSubBucketRecord.entries.pop_back();
            if constexpr (haveLocality)
            {
                --oldBucketRecord.entries.n;
            }

                // Note down the bucket for updating.
            bucketUpdates_.push_back(
                BucketUpdate{
                    .bucketIndex = oldBucketIndex,
                    .subBucketLabel = oldEntryRecord.bucketEntryIndex.subBucketLabel(),
                    .delta = -1
                });
        }
    }

    template <CBucketUpdater<BucketingScheme> UpdaterT>
    void
    doUpdateBucket(UpdaterT&& updater, gsl::index iBucket, std::span<SubBucketUpdate const> subBucketUpdates)
    {
        gsl_Expects(iBucket >= 0 && iBucket < std::ssize(bucketRecords_));

        auto& bucketRecord = bucketRecords_[iBucket];
        if (bucketRecords_[iBucket].entries.empty())
        {
                // Bucket is empty and can be removed.
                // If the bucket isn't at the end of the list, use a "castling" maneuver to remove it in  ν⋅𝒪(1) = 𝒪(ν)  steps.
            if (iBucket != std::ssize(bucketRecords_) - 1)
            {
                using std::swap;
                auto& swapBucketRef = *findBucketRef(bucketRecords_.back().label);
                swap(bucketRecords_[iBucket], bucketRecords_.back());
                swapBucketRef.bucketDataIndex = gsl::narrow_cast<Index>(iBucket);
                for (auto& bucketData2 : bucketRecords_)
                {
                    swap(bucketData2.bucketBucketData[iBucket], bucketData2.bucketBucketData.back());
                }
                iBucket = std::ssize(bucketRecords_) - 1;
            }
            if (!bucketRecord.flags.justAdded)
            {
                if constexpr (haveLocality)
                {
                    updater.onRemovingBucket(Bucket(bucketRecords_.begin() + iBucket), subBucketUpdates);
                }
                else
                {
                    updater.onRemovingBucket(Bucket(bucketRecords_.begin() + iBucket));
                }
            }
            auto bucketRefIt = findBucketRef(bucketRecords_[iBucket].label);
            orderedBucketRefs_.erase(bucketRefIt);
            for (gsl::index jBucket = 0; jBucket < iBucket; ++jBucket)
            {
                bucketRecords_[jBucket].bucketBucketData.pop_back();
            }
            bucketRecords_.pop_back();
        }
        else
        {
            if (bucketRecord.flags.justAdded)
            {
                if constexpr (haveLocality)
                {
                    updater.onBucketAdded(Bucket(bucketRecords_.begin() + iBucket), subBucketUpdates);
                }
                else
                {
                    updater.onBucketAdded(Bucket(bucketRecords_.begin() + iBucket));
                }
                bucketRecord.flags.justAdded = false;
            }
            else
            {
                if constexpr (haveLocality)
                {
                    updater.onBucketUpdated(Bucket(bucketRecords_.begin() + iBucket), subBucketUpdates);
                }
                else
                {
                    updater.onBucketUpdated(Bucket(bucketRecords_.begin() + iBucket));
                }
            }
        }
    }

private:
    template <template <typename> class EffectT>
    class Bucket_;
    template <template <typename> class EffectT>
    class SubBucket_
    {
        friend BucketingScheme;

        template <template <typename> class Effect2T> friend class Bucket_;
        template <template <typename> class Effect2T> friend class SubBucket_;

    private:
        SubBucketLabel label_;
        EffectT<SubBucketRecord>* subBucketRecord_;

        SubBucket_(SubBucketLabel const& _label, EffectT<SubBucketRecord>& _subBucketRecord)
            : label_(_label), subBucketRecord_(&_subBucketRecord)
        {
        }

        template <typename RecordIt>
        static auto
        subBucketRange(RecordIt firstRecord, gsl::index firstRecordIndex, gsl::dim num)
        {
            static_assert(haveLocality);
            static_assert(LocationDim == 1);  // for simplicity of implementation

            return std::views::iota(0, num)
                | std::views::transform(
                    [
                        firstRecord,
                        firstRecordIndex
                    ]
                    (gsl::index i)
                    {
                        return SubBucket_({ gsl::narrow_cast<LocationIndexT>(firstRecordIndex + i) }, firstRecord[i]);
                    });
        }
        static auto
        subBucketRange(EffectT<BucketRecord>& bucketRecord)
        {
            if constexpr (haveLocality)
            {
                static_assert(LocationDim == 1);  // for simplicity of implementation

                auto subBucketRecords = bucketRecord.entries.subBucketRecords.entries();
                static_assert(std::ranges::enable_borrowed_range<std::decay_t<decltype(subBucketRecords)>>);
                auto [firstIndex, lastIndex] = bucketRecord.entries.subBucketRecords.entryIndexRange();
                return subBucketRange(subBucketRecords.begin(), firstIndex[0], lastIndex[0] - firstIndex[0]);
            }
            else
            {
                return std::views::single(SubBucket_({ }, bucketRecord.entries));
            }
        }
        static auto
        subBucketRangeInRange(EffectT<BucketRecord>& bucketRecord, [[maybe_unused]] SubBucketLabel const& from, [[maybe_unused]] SubBucketLabel const& to)
        {
            if constexpr (haveLocality)
            {
                static_assert(LocationDim == 1);  // for simplicity of implementation

                auto subBucketRecords = bucketRecord.entries.subBucketRecords.entries();
                static_assert(std::ranges::enable_borrowed_range<std::decay_t<decltype(subBucketRecords)>>);
                auto [firstIndex, lastIndex] = bucketRecord.entries.subBucketRecords.entryIndexRange();
                auto [firstIndexInRange, lastIndexInRange] = bucketRecord.entries.subBucketRecords.entryIndexRangeInRange(from, to);

                return subBucketRange(subBucketRecords.begin() + (firstIndexInRange[0] - firstIndex[0]), firstIndexInRange[0], lastIndexInRange[0] - firstIndexInRange[0]);
            }
            else
            {
                return std::views::single(SubBucket_({ }, bucketRecord.entries));
            }
        }

    public:
        SubBucket_(detail::other_effect_t<SubBucket_, SubBucket_<std::type_identity_t>> const& rhs)
            : label_(rhs.label_), subBucketRecord_(rhs.subBucketRecord_)
        {
        }

        SubBucketLabel const&
        label() const
        {
            return label_;
        }
        std::span<Index const>
        entries() const
        {
            return subBucketRecord_->entries;
        }
        gsl::dim
        numEntries() const
        {
            return std::ssize(subBucketRecord_->entries);
        }
        EffectT<SubBucketData>&
        subBucketData()
        {
            return subBucketRecord_->subBucketData;
        }
        EffectT<SubBucketData> const&
        subBucketData() const
        {
            return subBucketRecord_->subBucketData;
        }

        friend void
        swap(SubBucket_& lhs, SubBucket_& rhs)
        {
            using std::swap;
            swap(lhs.label_, rhs.label_);
            swap(lhs.subBucketRecord_, rhs.subBucketRecord_);
        }
    };

    template <template <typename> class EffectT>
    class Bucket_
    {
        friend BucketingScheme;

        template <template <typename> class Effect2T> friend class Bucket_;

        using RecordIt = detail::effect_iterator_t<EffectT, std::vector<BucketRecord>>;

    private:
        RecordIt it_;

        explicit Bucket_(RecordIt _it)
            : it_(_it)
        {
        }
        auto
        inline allEntries() const
        {
            return it_->entries.allEntries();
        }
        auto subBucketImpl(SubBucketLabel const& label) const
        {
            return SubBucket_<EffectT>(label, it_->entries.subBucketRecords[label]);
        }

    public:
        Bucket_(detail::other_effect_t<Bucket_, Bucket_<std::type_identity_t>> const& rhs)
            : it_(rhs.it_)
        {
        }

        friend bool
        operator ==(Bucket_ const& lhs, Bucket_ const& rhs)
        {
            return lhs.it_ == rhs.it_;
        }

        BucketLabelT const&
        label() const
        {
            return it_->label;
        }

        EffectT<BucketData>&
        bucketData()
        {
            return it_->bucketData;
        }
        EffectT<BucketData> const&
        bucketData() const
        {
            return it_->bucketData;
        }

        bool
        areSubBucketsInitialized() const
        {
            return !it_->flags.noLocalityYet;
        }
        auto
        subBuckets() const
        {
            return SubBucket_<EffectT>::subBucketRange(*it_);
        }
        SubBucket_<EffectT>
        subBucket(SubBucketLabel label) const
        {
            if constexpr (haveLocality)
            {
                return subBucketImpl(label);
            }
            else
            {
                return SubBucket_<EffectT>(label, it_->entries);
            }
        }
        auto
        subBucketsInRange(SubBucketLabel first, SubBucketLabel last)
        {
            return SubBucket_<EffectT>::subBucketRangeInRange(*it_, first, last);
        }

        auto
        entries() const
        {
            if constexpr (haveLocality)
            {
                return allEntries();
            }
            else
            {
                return std::span<Index const>(it_->entries.entries);
            }
        }
        gsl::dim
        numEntries() const
        {
            return it_->entries.ssize();
        }

        friend void
        swap(Bucket_& lhs, Bucket_& rhs)
        {
            using std::swap;
            swap(lhs.it_, rhs.it_);
        }
    };

public:
    using ConstBucket = Bucket_<std::add_const_t>;
    using Bucket = Bucket_<std::type_identity_t>;
    using ConstSubBucket = SubBucket_<std::add_const_t>;
    using SubBucket = SubBucket_<std::type_identity_t>;

    BucketingScheme() = default;

    template <CBucketClassifier ClassifierT>
    explicit BucketingScheme(gsl::dim n, ClassifierT&& classifier)
    {
        if constexpr (haveLocality)
        {
            static_assert(CLocalBucketClassifier<ClassifierT, BucketingScheme>);
        }

        gsl_Expects(n >= 0 && n <= std::numeric_limits<Index>::max());

        std::vector<BucketLabelT> bucketLabels;

            // Add entries to bucket index cache.
        entryRecords_.reserve(n);
        bucketLabels.reserve(n);
        gsl::dim numInactiveEntries = 0;
        for (gsl::index i = 0; i != n; ++i)
        {
            std::optional<BucketLabelT> maybeLabel = classifier.classify(i);
            if (maybeLabel.has_value())
            {
                entryRecords_.push_back(EntryRecord{
                    .bucketLabel = *maybeLabel,
                    .bucketEntryIndex = BucketEntryIndex(std::true_type{ }, 0, { })  // placeholder index
                });
                detail::push_back_if_neq_last(bucketLabels, *maybeLabel);
            }
            else
            {
                entryRecords_.push_back(EntryRecord{
                    .bucketLabel = { },
                    .bucketEntryIndex = BucketEntryIndex(std::false_type{ }, 0)  // placeholder index
                });
                ++numInactiveEntries;
            }
        }

            // Extract sorted list of unique bucket labels.
        std::ranges::sort(bucketLabels);
        auto last = std::unique(bucketLabels.begin(), bucketLabels.end());
        bucketLabels.resize(last - bucketLabels.begin());

            // Add buckets.
        orderedBucketRefs_.reserve(bucketLabels.size());
        bucketRecords_.reserve(bucketLabels.size());
        for (gsl::index iB = 0, nB = std::ssize(bucketLabels); iB != nB; ++iB)
        {
            auto const& label = bucketLabels[iB];
            orderedBucketRefs_.push_back(BucketRef{
                .label = label,
                .bucketDataIndex = gsl::narrow_cast<Index>(iB)
            });
            bucketRecords_.push_back(BucketRecord{
                .bucketBucketData = std::vector<BucketBucketData>(nB),
                .label = label
            });
        }

            // Add entries to buckets, and keep track of inactive entries (in reverse order).
        inactiveEntries_.resize(numInactiveEntries);
        gsl::index ii = numInactiveEntries - 1;
        for (gsl::index i = 0; i != n; ++i)
        {
            auto& entryRecord = entryRecords_[i];
            if (entryRecord.bucketEntryIndex.active())
            {
                gsl::index bucketIndex = findBucketRef(entryRecord.bucketLabel)->bucketDataIndex;
                auto& bucketRecord = bucketRecords_[bucketIndex];
                auto newBucketEntryIndex = gsl::narrow_cast<Index>(bucketRecord.entries.ssize());
                if constexpr (haveLocality)
                {
                    bucketRecord.entries.subBucketRecords.obtain({ }).entries.push_back(gsl::narrow_cast<Index>(i));
                    ++bucketRecord.entries.n;
                }
                else
                {
                    bucketRecord.entries.entries.push_back(gsl::narrow_cast<Index>(i));
                }
                entryRecord.bucketEntryIndex = BucketEntryIndex(std::true_type{ }, newBucketEntryIndex, { });
            }
            else
            {
                entryRecord.bucketEntryIndex = BucketEntryIndex(std::false_type{ }, ii);
                inactiveEntries_[ii] = gsl::narrow_cast<Index>(i);
                --ii;
            }
        }
    }

    auto
    buckets()
    {
        //return std::views::iota(bucketRecords_.begin(), bucketRecords_.end())
        return orderedBucketRefs_
             | std::views::transform(
                   //[]<typename T>(T&& it)
                   //{
                   //    return Bucket(std::forward<T>(it));
                   //});
                   [firstBucketIt = bucketRecords_.begin()]
                   (BucketRef const& bucketRef)
                   {
                       return Bucket(firstBucketIt + bucketRef.bucketDataIndex);
                   });
    }
    auto
    buckets() const
    {
        //return std::views::iota(bucketRecords_.begin(), bucketRecords_.end())
        return orderedBucketRefs_
             | std::views::transform(
                   //[]<typename T>(T&& it)
                   //{
                   //    return ConstBucket(std::forward<T>(it));
                   //});
                   [firstBucketIt = bucketRecords_.begin()]
                   (BucketRef const& bucketRef)
                   {
                       return ConstBucket(firstBucketIt + bucketRef.bucketDataIndex);
                   });
    }

    bool
    isEntryActive(gsl::index entryIndex)
    {
        gsl_Expects(entryIndex >= 0 && entryIndex < std::ssize(entryRecords_));
    
        return entryRecords_[entryIndex].active();
    }

    std::optional<Bucket>
    searchBucketOfEntry(gsl::index entryIndex)
    {
        gsl_Expects(entryIndex >= 0 && entryIndex < std::ssize(entryRecords_));
    
        auto& entryRecord = entryRecords_[entryIndex];
        if (!entryRecord.active()) return std::nullopt;
        auto& bucketRef = *findBucketRef(entryRecord.bucketLabel);
        return Bucket(bucketRecords_.begin() + bucketRef.bucketDataIndex);
    }
    std::optional<ConstBucket>
    searchBucketOfEntry(gsl::index entryIndex) const
    {
        gsl_Expects(entryIndex >= 0 && entryIndex < std::ssize(entryRecords_));
    
        auto& entryRecord = entryRecords_[entryIndex];
        if (!entryRecord.active()) return std::nullopt;
        auto& bucketRef = *findBucketRef(entryRecord.bucketLabel);
        return ConstBucket(bucketRecords_.begin() + bucketRef.bucketDataIndex);
    }
    Bucket
    findBucketOfEntry(gsl::index entryIndex)
    {
        return searchBucketOfEntry(entryIndex).value();
    }
    ConstBucket
    findBucketOfEntry(gsl::index entryIndex) const
    {
        return searchBucketOfEntry(entryIndex).value();
    }

    std::span<Index const>
    inactiveEntries() const
    {
        return inactiveEntries_;
    }

    BucketBucketData&
    bucketBucketData(Bucket bucket1, Bucket bucket2)
    {
        gsl::index bucket2Index = bucket2.it_ - bucketRecords_.begin();
        return bucket1.it_->bucketBucketData[bucket2Index];
    }
    BucketBucketData const&
    bucketBucketData(ConstBucket bucket1, ConstBucket bucket2) const
    {
        gsl::index bucket2Index = bucket2.it_ - bucketRecords_.begin();
        return bucket1.it_->bucketBucketData[bucket2Index];
    }

    template <CBucketClassifier ClassifierT, CBucketUpdater<BucketingScheme> UpdaterT = DefaultBucketUpdater<BucketingScheme>>
    void
    updateEntries(std::span<gsl::index const> entryIndices, ClassifierT&& classifier, UpdaterT&& updater = { })
    {
        if constexpr (haveLocality)
        {
            static_assert(CLocalBucketClassifier<ClassifierT, BucketingScheme>);
        }

            // Defend against reentrancy.
        gsl_Assert(!updating_);
        updating_ = true;
        auto _ = gsl::finally([&] { updating_ = false; });

            // Update bucket assignments of given particles.
        for (gsl::index i : entryIndices)
        {
            doUpdateEntry(updater, classifier, i);
        }

            // The relational comparison operators compare the bucket label in reverse order, and we therefore
            // process buckets in descending order of bucket indices. Therefore, if a bucket is removed, and to
            // this end swapped with the last bucket with the "castling" maneuver, we can be sure that any updates
            // for the swapped bucket have already been processed, and hence no bucket indices are invalidated.
            // We then coalesce sub-bucket updates and notify the updater.
        std::sort(bucketUpdates_.begin(), bucketUpdates_.end());
        auto pos = bucketUpdates_.begin();
        auto lastBucketIt = bucketUpdates_.end();
        while (pos != lastBucketIt)
        {
                // Find range of sub-bucket updates in bucket.
            auto nextBucketIt = pos;
            do
            {
                ++nextBucketIt;
            } while (nextBucketIt != lastBucketIt && nextBucketIt->bucketIndex == pos->bucketIndex);

                // Coalesce sub-bucket updates.
            auto lsubBucketUpdates = std::span<SubBucketUpdate const>{ };
            if constexpr (haveLocality)
            {
                lsubBucketUpdates = coalesceSubBucketUpdates(subBucketUpdates_, pos, nextBucketIt);
            }

                // Update bucket.
            doUpdateBucket(updater, pos->bucketIndex, lsubBucketUpdates);

            pos = nextBucketIt;
        }
        bucketUpdates_.clear();
    }

    template <CLocalBucketClassifier<BucketingScheme> ClassifierT>
    void
    updateLocationSubBuckets(Bucket bucket, ClassifierT&& classifier)
    {
        if constexpr (haveLocality)
        {
            auto newEntries = LocalEntries{
                .n = gsl::narrow_cast<Dim>(bucket.numEntries())
            };
            auto& oldEntries = bucket.it_->entries;
            for (auto const& subBucketEntries : oldEntries.subBucketRecords.entries())
            {
                for (gsl::index i : subBucketEntries.entries)
                {
                    auto subBucketLabel = classifier.subclassify(bucket, i);
                    newEntries.subBucketRecords.obtain(subBucketLabel).entries.push_back(gsl::narrow_cast<Index>(i));
                }
            }
            for (auto&& [subBucketLabel, subBucketEntries] : newEntries.subBucketRecords.indexedEntries())
            {
                for (gsl::index ii = 0, ni = std::ssize(subBucketEntries.entries); ii != ni; ++ii)
                {
                    gsl::index i = subBucketEntries.entries[ii];
                    entryRecords_[i].bucketEntryIndex = LocalBucketEntryIndex(std::true_type{ }, ii, subBucketLabel);
                }
            }
            oldEntries = std::move(newEntries);
            bucket.it_->flags.noLocalityYet = false;
        }
    }
};


} // namespace rpmc


#endif // INCLUDED_RPMC_TOOLS_BUCKETING_HPP_
