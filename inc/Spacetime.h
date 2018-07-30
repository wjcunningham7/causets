/////////////////////////////
//(C) Will Cunningham 2017 //
//         DK Lab          //
// Northeastern University //
/////////////////////////////

#ifndef SPACETIME_H_
#define SPACETIME_H_

#include <algorithm>
#include <stdio.h>
#include <FastBitset.h>

inline uint64_t hex_to_u64(const char *hex64)
{
	uint64_t value = 0;
	long upper;
	char hex[17];

	unsigned int hexlen = strlen(hex64);
	if (hexlen < sizeof(hex)) {
		strcpy(hex, hex64);
		if (hexlen > 8) {
			sscanf(&hex[hexlen-8], "%x", (unsigned*)&value);
			hex[hexlen-8] = 0;
			sscanf(hex, "%x", (unsigned*)&upper);
			value |= ((uint64_t)upper << 32);
		} else
			sscanf(hex, "%x", (unsigned*)&value);
	}

	return value;
}

class Spacetime
{
public:
	static constexpr const char *stdims[] = { "2", "3", "4", "5" };
	static constexpr const char *manifolds[] = { "Minkowski", "De_Sitter", "Anti_de_Sitter", "Dust", "FLRW", "Hyperbolic", "Polycone" };
	static constexpr const char *regions[] = { "Slab", "Slab_T1", "Slab_T2", "Slab_TS", "Slab_S1", "Slab_S2", "Slab_N1", "Slab_N2", "Slab_N3", "Half_Diamond", "Diamond", "Saucer_T", "Saucer_S", "Triangle_T", "Triangle_S", "Triangle_N", "Cube" };
	static constexpr const char *curvatures[] = { "Positive", "Flat", "Negative" };
	static constexpr const char *symmetries[] = { "None", "Temporal" };

	static const unsigned int nstdims = sizeof(stdims) / sizeof(char*);
	static const unsigned int nmanifolds = sizeof(manifolds) / sizeof(char*);
	static const unsigned int nregions = sizeof(regions) / sizeof(char*);
	static const unsigned int ncurvatures = sizeof(curvatures) / sizeof(char*);
	static const unsigned int nsymmetries = sizeof(symmetries) / sizeof(char*);
	static const unsigned int stsize = nstdims + nmanifolds + nregions + ncurvatures + nsymmetries;

	Spacetime()
	{
		spacetime = new FastBitset(stsize);
		create_masks();
	}

	Spacetime(const char *stdim, const char *manifold, const char *region, const char *curvature, const char *symmetry)
	{
		spacetime = new FastBitset(stsize);
		create_masks();
		set_spacetime(stdim, manifold, region, curvature, symmetry);
	}

	Spacetime(const Spacetime &other)
	{
		create_masks();
		spacetime = new FastBitset(stsize);
		other.spacetime->clone(*spacetime);
	}

	Spacetime& operator= (const Spacetime &other)
	{
		FastBitset *_spacetime = new FastBitset(stsize);
		other.spacetime->clone(*_spacetime);
		delete spacetime;
		spacetime = _spacetime;
		return *this;
	}

	~Spacetime()
	{
		delete spacetime;
		delete stdim_mask;
		delete manifold_mask;
		delete region_mask;
		delete curvature_mask;
		delete symmetry_mask;
		delete workspace;
	}

	inline bool operator==(Spacetime const &other)
	{
		return *spacetime == *(other.spacetime);
	}

	void set_spacetime(const char *stdim, const char *manifold, const char *region, const char *curvature, const char *symmetry)
	{
		spacetime->reset();
		spacetime->set(std::distance(stdims, std::find(stdims, stdims + nstdims, std::string(stdim))));
		spacetime->set(std::distance(manifolds, std::find(manifolds, manifolds + nmanifolds, std::string(manifold))) + nstdims);
		spacetime->set(std::distance(regions, std::find(regions, regions + nregions, std::string(region))) + nstdims + nmanifolds);
		spacetime->set(std::distance(curvatures, std::find(curvatures, curvatures + ncurvatures, std::string(curvature))) + nstdims + nmanifolds + nregions);
		spacetime->set(std::distance(symmetries, std::find(symmetries, symmetries + nsymmetries, std::string(symmetry))) + nstdims + nmanifolds + nregions + ncurvatures);
	}

	int get_stdim() const
	{
		stdim_mask->clone(*workspace);
		workspace->setIntersection(*spacetime);
		return workspace->next_bit();
	}

	bool stdimIs(const char *stdim) const
	{
		return !strcmp(stdims[this->get_stdim()], stdim);
	}

	int get_manifold() const
	{
		manifold_mask->clone(*workspace);
		workspace->setIntersection(*spacetime);
		return workspace->next_bit() - nstdims;
	}

	bool manifoldIs(const char *manifold) const
	{
		return !strcmp(manifolds[this->get_manifold()], manifold);
	}

	int get_region() const
	{
		region_mask->clone(*workspace);
		workspace->setIntersection(*spacetime);
		return workspace->next_bit() - nstdims - nmanifolds;
	}

	bool regionIs(const char *region) const
	{
		return !strcmp(regions[this->get_region()], region);
	}

	int get_curvature() const
	{
		curvature_mask->clone(*workspace);
		workspace->setIntersection(*spacetime);
		return workspace->next_bit() - nstdims - nmanifolds - nregions;
	}

	bool curvatureIs(const char *curvature) const
	{
		return !strcmp(curvatures[this->get_curvature()], curvature);
	}

	int get_symmetry() const
	{
		symmetry_mask->clone(*workspace);
		workspace->setIntersection(*spacetime);
		return workspace->next_bit() - nstdims - nmanifolds - nregions - ncurvatures;
	}

	bool symmetryIs(const char *symmetry) const
	{
		return !strcmp(symmetries[this->get_symmetry()], symmetry);
	}

	bool spacetimeIs(const char *stdim, const char *manifold, const char *region, const char *curvature, const char *symmetry) const
	{
		return this->stdimIs(stdim) && this->manifoldIs(manifold) && this->regionIs(region) && this->curvatureIs(curvature) && this->symmetryIs(symmetry);
	}

	const char* toHexString() const
	{
		char buffer[stsize];
		for (uint64_t i = 0; i < stsize / 64; i++)
			snprintf(&buffer[i*16], 16, "%lx", spacetime->readBlock(i));

		std::ostringstream s;
		s << buffer;

		return s.str().c_str();
	}

	void fromHexString(char *hs)
	{
		std::stringstream s;
		s << std::hex << hs;
		for (uint64_t i = 0; i <= s.str().size() >> 16; i++)
			this->spacetime->writeBlock(hex_to_u64(&hs[i<<16], i);
	}

private:
	FastBitset *spacetime;
	FastBitset *stdim_mask;
	FastBitset *manifold_mask;
	FastBitset *region_mask;
	FastBitset *curvature_mask;
	FastBitset *symmetry_mask;
	FastBitset *workspace;

	void create_masks()
	{
		unsigned int i = 0;
		stdim_mask = new FastBitset(stsize);
		while (i < nstdims)
			stdim_mask->set(i++);

		manifold_mask = new FastBitset(stsize);
		while (i < nstdims + nmanifolds)
			manifold_mask->set(i++);

		region_mask = new FastBitset(stsize);
		while (i < nstdims + nmanifolds + nregions)
			region_mask->set(i++);

		curvature_mask = new FastBitset(stsize);
		while (i < nstdims + nmanifolds + nregions + ncurvatures)
			curvature_mask->set(i++);

		symmetry_mask = new FastBitset(stsize);
		while (i < stsize)
			symmetry_mask->set(i++);

		workspace = new FastBitset(stsize);
	}
};

#endif

