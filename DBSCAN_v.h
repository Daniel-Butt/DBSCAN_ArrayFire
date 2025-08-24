#pragma once

#include <arrayfire.h>
#include <vector>
#include <queue>
#include <cstdint>
#include <iostream>
#include <limits>
#include <algorithm>

class DBSCAN {
private:
    static constexpr int NOISE = -1;
    static constexpr int UNASSIGNED = -2;

public:
    float eps;
    int minPts;
    int numClusters;

    DBSCAN(float eps, int minPts) : eps(eps), minPts(minPts), numClusters(0) {}

    std::vector<int> fit(const af::array& X) {
        const int N = X.dims(0);
        numClusters = 0;

		// Compute adjacency matrix
        af::array D = af::matmul(X, X, AF_MAT_NONE, AF_MAT_TRANS) > eps;

		// Identify core points
        af::array C = af::sum(D, 1) >= minPts;

        // Convert dense adjacency matrix to CSR sparse format
        af::array S = af::sparse(D.as(f32), AF_STORAGE_CSR);
        af::array rowIdx = af::sparseGetRowIdx(S);
        af::array colIdx = af::sparseGetColIdx(S);

		// Transfer to host (GPU -> CPU)
        std::vector<int> csr_row_ptr(rowIdx.elements());
        std::vector<int> csr_col_idx(colIdx.elements());
        std::vector<byte> core(N);
        
        rowIdx.host(csr_row_ptr.data());
        colIdx.host(csr_col_idx.data());
        C.host(core.data());

        int cluster_id = -1;
        std::vector<int> labels(N, UNASSIGNED);
        std::queue<int> q;

		//Core points
        for (int i = 0; i < N; i++) {
            if (!core[i] || labels[i] != UNASSIGNED) continue;

            cluster_id++;
            labels[i] = cluster_id;
            q.push(i);

            while (!q.empty()) {
                int u = q.front(); q.pop();

                // Iterate through neighbors of u using CSR format
                for (int idx = csr_row_ptr[u]; idx < csr_row_ptr[u + 1]; idx++) {
                    int v = csr_col_idx[idx];
                    if (!core[v] || labels[v] != UNASSIGNED) continue;

                    labels[v] = cluster_id;
                    q.push(v);
                }
            }
        }
        numClusters = cluster_id + 1;

        // Border points
        if (numClusters > 0) {
            for (int i = 0; i < N; i++) {
                if (core[i]) continue;

                // Iterate through neighbors of i using CSR format
                for (int idx = csr_row_ptr[i]; idx < csr_row_ptr[i + 1]; idx++) {
                    int v = csr_col_idx[idx];
                    if (!core[v]) continue;

                    labels[i] = labels[v];
                    break;
                }

                if (labels[i] == UNASSIGNED) labels[i] = NOISE;
            }
        }
        else {
            // No core points: every point is noise
            std::fill(labels.begin(), labels.end(), NOISE);
        }

        return labels;
    }
};
