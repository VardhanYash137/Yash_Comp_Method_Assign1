#include <stdio.h>
#include <gsl/gsl_linalg.h>

int main() {
    // Define a sample matrix
    double data[] = {2.0, 1.0, 1.0,
                     4.0, -6.0, 0.0,
                     -2.0, 7.0, 2.0};

    // Create a GSL matrix view
    gsl_matrix_view m = gsl_matrix_view_array(data, 3, 3);

    // Perform LU decomposition
    gsl_permutation *p = gsl_permutation_alloc(3); // permutation vector
    int signum; // sign of the permutation

    gsl_linalg_LU_decomp(&m.matrix, p, &signum);

    // Print the original matrix
    printf("Original Matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            printf("%g ", gsl_matrix_get(&m.matrix, i, j));
        }
        printf("\n");
    }

    // Print the LU decomposition matrix
    printf("\nLU Decomposition:\n");
    printf("LU Matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            printf("%g ", gsl_matrix_get(&m.matrix, i, j));
        }
        printf("\n");
    }

    // Print the permutation matrix
    printf("\nPermutation Matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (j == gsl_permutation_get(p, i)) {
                printf("1 ");
            } else {
                printf("0 ");
            }
        }
        printf("\n");
    }

    // Free memory
    gsl_permutation_free(p);

    return 0;
}
