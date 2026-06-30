#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX 1000

int comparisons = 0;
int swaps = 0;

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
    swaps++;
}

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            comparisons++;
            if (arr[j] > arr[j+1]) {
                swap(&arr[j], &arr[j+1]);
            }
        }
    }
}

void selection_sort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        int min_idx = i;
        for (int j = i+1; j < n; j++) {
            comparisons++;
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            swap(&arr[min_idx], &arr[i]);
        }
    }
}

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0) {
            comparisons++;
            if (arr[j] > key) {
                arr[j+1] = arr[j];
                swaps++;
                j--;
            } else {
                break;
            }
        }
        arr[j+1] = key;
    }
}

void merge(int arr[], int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;
    int L[n1], R[n2];

    for (int i = 0; i < n1; i++) L[i] = arr[l+i];
    for (int j = 0; j < n2; j++) R[j] = arr[m+1+j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        comparisons++;
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
            swaps++;
        }
    }
    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

void merge_sort(int arr[], int l, int r) {
    if (l < r) {
        int m = l + (r-l)/2;
        merge_sort(arr, l, m);
        merge_sort(arr, m+1, r);
        merge(arr, l, m, r);
    }
}

void print_array(int arr[], int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int n, choice;
    printf("enter number of elements: ");
    scanf("%d", &n);

    int arr[n];
    srand(time(0));
    for (int i = 0; i < n; i++) {
        arr[i] = (rand() % 1000) + 1;
    }

    printf("original array:\n");
    print_array(arr, n);

    printf("choose sorting algorithm:\n");
    printf("1. bubble sort\n2. selection sort\n3. insertion sort\n4. merge sort\n");
    scanf("%d", &choice);

    comparisons = 0;
    swaps = 0;

    switch(choice) {
        case 1: bubble_sort(arr, n); break;
        case 2: selection_sort(arr, n); break;
        case 3: insertion_sort(arr, n); break;
        case 4: merge_sort(arr, 0, n-1); break;
        default: printf("invalid choice\n"); return 0;
    }

    printf("sorted array:\n");
    print_array(arr, n);

    printf("total comparisons: %d\n", comparisons);
    printf("total swaps: %d\n", swaps);

    return 0;
}
