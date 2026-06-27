#include <stdio.h>

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void max_heap(int arr[], int n, int i) {
    int largest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < n && arr[left] > arr[largest])
        largest = left;

    if (right < n && arr[right] > arr[largest])
        largest = right;

    if (largest != i) {
        swap(&arr[i], &arr[largest]);
        max_heap(arr, n, largest);
    }
}

void min_heap(int arr[], int n, int i) {
    int smallest = i;
    int left = 2*i + 1;
    int right = 2*i + 2;

    if (left < n && arr[left] < arr[smallest])
        smallest = left;

    if (right < n && arr[right] < arr[smallest])
        smallest = right;

    if (smallest != i) {
        swap(&arr[i], &arr[smallest]);
        min_heap(arr, n, smallest);
    }
}

void build_max(int arr[], int n) {
    for (int i = n/2 - 1; i >= 0; i--)
        max_heap(arr, n, i);
}

void build_min(int arr[], int n) {
    for (int i = n/2 - 1; i >= 0; i--)
        min_heap(arr, n, i);
}

void print(int arr[], int n) {
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");
}

int main() {
    int arr[] = {9, 4, 7, 1, -2, 6, 5};
    int n = sizeof(arr)/sizeof(arr[0]);

    printf("original array:\n");
    print(arr, n);

    build_max(arr, n);
    printf("max heap:\n");
    print(arr, n);

    int arr2[] = {9, 4, 7, 1, -2, 6, 5};
    build_min(arr2, n);
    printf("min heap:\n");
    print(arr2, n);

    return 0;
}