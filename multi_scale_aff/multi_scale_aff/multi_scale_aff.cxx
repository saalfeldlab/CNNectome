#include "multi_scale_aff.hxx"
#include <unordered_map>
#include <array>
#include <vector>

typedef std::array<size_t, 3> shape;
typedef std::array<size_t, 4> aff_shape;
typedef std::unordered_map<int,double> histogram;

size_t get_strided_idx(const int coord_ax0, const int coord_ax1, const int coord_ax2, const shape &strides){
    return coord_ax0*strides[0]+coord_ax1*strides[1]+coord_ax2*strides[2];
}

size_t get_strided_idx(const int coord_ax0, const int coord_ax1, const int coord_ax2,
                       const int coord_ax3, const aff_shape &strides){
    return coord_ax0*strides[0]+coord_ax1*strides[1]+coord_ax2*strides[2]+coord_ax3*strides[3];
}


void compute_histogram(const int64_t* labels, const int coord_ax0, const int coord_ax1, const int coord_ax2,
                       const int blocking_ax0, const int blocking_ax1, const int blocking_ax2,
                       const shape &strides, histogram &histo){
     for (size_t ax0 = 0; ax0< blocking_ax0; ax0++){
        for (size_t ax1 = 0; ax1< blocking_ax1; ax1++){
            size_t idx = get_strided_idx(coord_ax0+ax0, coord_ax1+ax1, coord_ax2, strides);
            for (size_t ax2 = 0; ax2< blocking_ax2; ax2++){
                  int64_t label = *(labels+idx+ax2);
                  auto map_it = histo.find(label);
                  if (map_it == histo.end()) {
                    histo[label] = 1.;
                  } else {
                    map_it->second += 1;
                  }

            }
        }
    }
}

double compute_single_affinity(const histogram& center, const histogram& neighbor, const int num_vox_per_block){
    double aff = 0.;
    const double vox_squared = num_vox_per_block * num_vox_per_block;
    for (auto it = center.begin(); it != center.end(); it++){
        auto neigh_it = neighbor.find(it->first);
        if (neigh_it != neighbor.end())
            aff +=  it->second * neigh_it->second / vox_squared;
    }
    return aff;
}


void compute_multi_scale_affinities_impl(
    const int64_t* labels,
    double* affinities,
    const int blocking_ax0,
    const int blocking_ax1,
    const int blocking_ax2,
    const int labels_shape_ax0,
    const int labels_shape_ax1,
    const int labels_shape_ax2
) {
    const int blocks_per_ax0 = labels_shape_ax0/blocking_ax0;
    const int blocks_per_ax1 = labels_shape_ax1/blocking_ax1;
    const int blocks_per_ax2 = labels_shape_ax2/blocking_ax2;

    shape strides;
    strides[0]=labels_shape_ax1*labels_shape_ax2;
    strides[1]=labels_shape_ax2;
    strides[2]=1;


    shape block_strides;
    block_strides[0]=blocking_ax1*blocking_ax2;
    block_strides[1]=blocks_per_ax2;
    block_strides[2]=1;


    aff_shape aff_strides;
    aff_strides[0]=blocking_ax0*blocking_ax1*blocking_ax2;
    aff_strides[1]=blocking_ax1 * blocking_ax2;
    aff_strides[2]=blocking_ax2;
    aff_strides[3]=1;


    int num_blocks = blocks_per_ax0 * blocks_per_ax1 * blocks_per_ax2;
    int num_vox_per_block = blocking_ax0 * blocking_ax1 * blocking_ax2;
    std::vector<histogram> histograms(num_blocks);
    for (size_t ax0 = 0; ax0 < blocks_per_ax0; ax0++){
        for (size_t ax1 = 0; ax1 < blocks_per_ax1; ax1++){
            for (size_t ax2 = 0; ax2 < blocks_per_ax2; ax2++){
                int coord_ax0 = ax0 * blocking_ax0;
                int coord_ax1 = ax1 * blocking_ax1;
                int coord_ax2 = ax2 * blocking_ax2;

                size_t block_idx = get_strided_idx(ax0, ax1, ax2, block_strides);
                histogram &histo = histograms[block_idx];
                compute_histogram(labels, coord_ax0, coord_ax1, coord_ax2, blocking_ax0, blocking_ax1, blocking_ax2,
                strides, histo);
            }
        }
    }


    for (size_t ax0 = 1; ax0 < blocks_per_ax0; ax0++){
        for (size_t ax1 = 1; ax1 < blocks_per_ax1 ; ax1++){
            for (size_t ax2 = 1; ax2 < blocks_per_ax2; ax2++){
                size_t block_idx = get_strided_idx(ax0, ax1, ax2, block_strides);

                size_t block_idx_0 = get_strided_idx(ax0-1, ax1, ax2, block_strides);
                size_t block_idx_1 = get_strided_idx(ax0, ax1-1, ax2, block_strides);
                size_t block_idx_2 = get_strided_idx(ax0, ax1, ax2-1, block_strides);

                size_t aff_idx_0 = get_strided_idx(0, ax0-1, ax1, ax2, aff_strides);
                size_t aff_idx_1 = get_strided_idx(1, ax0, ax1-1, ax2, aff_strides);
                size_t aff_idx_2 = get_strided_idx(2, ax0, ax1, ax2-1, aff_strides);

                *(affinities+aff_idx_0) = compute_single_affinity(histograms[block_idx], histograms[block_idx_0],
                                                                  num_vox_per_block);

                *(affinities+aff_idx_1) = compute_single_affinity(histograms[block_idx], histograms[block_idx_1],
                                                                  num_vox_per_block);

                *(affinities+aff_idx_2) = compute_single_affinity(histograms[block_idx], histograms[block_idx_2],
                                                                  num_vox_per_block);





            }
        }
    }
}