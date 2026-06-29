#include <stdio.h>
#include <stdlib.h>

struct Node {
    int data;
    struct Node* next;
    struct Node* prev;
};

struct Node* createnode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    newNode->prev = NULL;
    return newNode;
}

void insertend(struct Node** head, int data) {
    struct Node* newNode = createnode(data);
    if (*head == NULL) {
        *head = newNode;
        return;
    }
    struct Node* temp = *head;
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = newNode;
    newNode->prev = temp;
    }
//necessary to seacrh for the required node 
    struct Node* searchnode(struct Node* head, int key) {
        struct Node* temp = head;
        while (temp != NULL) {
            if (temp->data == key) return temp;
            temp = temp->next;
        }
        return NULL;
    }
void insertaftervalue(struct Node* head, int key, int data) {
    struct Node* prevNode = searchnode(head, key);
    if (prevNode == NULL) {
        printf("Node with value %d not found\n", key);
        return;
    }
    struct Node* newNode = createnode(data);
    newNode->next = prevNode->next;
    newNode->prev = prevNode;
    if (prevNode->next != NULL) {
        prevNode->next->prev = newNode;
    }
    prevNode->next = newNode;//reassingning values
}
void deletebyvalue(struct Node** head, int key) {
    struct Node* delNode = searchnode(*head, key);
    if (*head == NULL || delNode == NULL) return;
    if (*head == delNode) {
        *head = delNode->next;
    }
    if (delNode->next != NULL) {
        delNode->next->prev = delNode->prev;
    }
    if (delNode->prev != NULL) {
        delNode->prev->next = delNode->next;
    }
    free(delNode);//use of free to prevent dangling pointer and free memory
}
void print(struct Node* head) {
    struct Node* temp = head;
    while (temp != NULL) {
        printf("%d ", temp->data);
        temp = temp->next;
    }
    printf("\n");
}
int main() {
    struct Node* head = NULL;// **represents pointer to pointer 

    insertend(&head, 30);
    insertend(&head, 20);
    insertend(&head, 40);

    printf("Initial list: ");
    print(head);

    insertaftervalue(head, 30, 10); // insert 31 after node with value 30
    print(head);

    deletebyvalue(&head, 20); // delete node with value 20
    print(head);

    return 0;
}




