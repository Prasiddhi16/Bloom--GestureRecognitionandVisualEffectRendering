Documentation for program 4: Doubly Linked List Insert After and Delete  

(A)Data Structures Defined  
- A structure Node is defined with three fields  
- int data stores the value of the node  
- struct Node* next is a self refrential pointer to the next node  
- struct Node* prev is a self refrential pointer to the previous node  
- A pointer head is used to track the starting node of the list  

(B) Functions  

1:createNode(int data)  
-allocates memory for a new node  
-initializes data and sets next and prev to NULL  
-returns the new node  

2:insertEnd(struct Node** head, int data)  
-inserts a new node at the end of the list  
-if the list is empty, assigns the new node as head  
-otherwise traverses to the last node and links the new node at the end  

3:searchNode(struct Node* head, int key)  
-traverses the list from head to end  
-compares each node’s data with the key  
-returns the pointer to the node if found or NULL if not found  

4:insertAfterValue(struct Node* head, int key, int data)  
-searches for the node with the given key using searchNode  
-if found, creates a new node and places it after the matched node  
-updates next and prev pointers to maintain the list structure  
-prints message if the key node is not found  

5:deleteByValue(struct Node** head, int key)  
-searches for the node with the given key  
-if found, adjusts the links of the previous and next nodes to bypass the node  
-if the node to be deleted is head, updates head to the next node  
-frees the memory of the deleted node  

6:print(struct Node* head)  
-traverses the list from head to end  
-prints the data of each node in forward order  

(C) The main() function  
-initializes the list with head = NULL  
-calls insertEnd to add nodes with values 30, 20, 40  
-prints the initial list  
-calls insertAfterValue to insert 10 after the node with value 30  
-prints the updated list  
-calls deleteByValue to delete the node with value 20  
-prints the final list  

(D)Output  

Initial list: 30 20 40  
30 10 20 40  
30 10 40  
