 Documentataion for program 3: Doubly Linked List Reverse Traversal
 
 (A) Data Structures Defined
- A structure Node is defined with three fields:
- int data stores the value of the node
- struct Node* next is a self‑referential pointer to the next node
- struct Node* prev is a self‑referential pointer to the previous node
- A pointer head is used to track the starting node of the list

(B) Functions
1: createNode(int data)
- Dynamically allocates memory for a new node
- Initializes data, sets next and prev to NULL
- Returns the newly created node
2: insertEnd(struct Node** head, int data)
- Inserts a new node at the end of the doubly linked list
- If the list is empty, assigns the new node as head
- Otherwise, traverses to the last node and links the new node at the end
- Updates both next and prev pointers accordingly
3: traverseForward(struct Node* head)
- Traverses the list from head to the end using next pointers
- Prints the data of each node in forward order
4: traverseReverse(struct Node* head)
- Traverses the list to the last node using next pointers
- Then moves backward using prev pointers
- Prints the data of each node in reverse order


(C) The main() Function
- Initializes the list with head = NULL
- Calls insertEnd to add nodes with values 10, 20, 30, 40
- Calls traverseForward to display the list in forward order
- Calls traverseReverse to display the list in reverse order

(D) Output
Forward Traversal: 10 20 30 40
Reverse Traversal: 40 30 20 10



