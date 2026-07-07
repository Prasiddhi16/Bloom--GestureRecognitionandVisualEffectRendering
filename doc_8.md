Program: Sorting with User Choice
(A) Data Structures Defined
- an integer array arr[] is defined to hold the randomly generated numbers.
- n is the size of the array, provided by the user at run time.
- comparisons is an integer counter to track the number of comparisons performed.
- swaps is an integer counter to track the number of swaps performed.
(B) Functions
1: swap(int *a, int *b)
- swaps two integer values using a temporary variable.
- increments the swap counter.
2: bubble_sort(int arr[], int n)
- compares adjacent elements and swaps if necessary.
- repeats until the array is sorted.
- counts comparisons and swaps.
3: selection_sort(int arr[], int n)
- finds the minimum element in the unsorted part of the array.
- swaps it with the first element of the unsorted part.
- counts comparisons and swaps.
4: insertion_sort(int arr[], int n)
- inserts each element into its correct position in the sorted part of the array.
- shifts elements as needed.
- counts comparisons and swaps.
5: merge(int arr[], int l, int m, int r)
- merges two sorted subarrays into a single sorted array.
- counts comparisons and swaps during merging.
6: merge_sort(int arr[], int l, int r)
- recursively divides the array into halves.
- calls merge to combine them.
7: print_array(int arr[], int n)
- prints the array elements in sequence.

(C) The main() function
- asks the user for the number of elements n.
- generates n random integers in the range [1, 1000].
- prints the original array.
- asks the user to choose a sorting algorithm: bubble sort, selection sort, insertion sort, or merge sort.
- calls the chosen sorting function.
- prints the sorted array.
- displays the total number of comparisons and swaps performed.
D) Output
enter number of elements: 10
original array:
123 876 45 999 302 678 12 456 789 234
choose sorting algorithm:
1. bubble sort
2. selection sort
3. insertion sort
4. merge sort
2
sorted array:
12 45 123 234 302 456 678 789 876 999
total comparisons: 45
total swaps: 9




