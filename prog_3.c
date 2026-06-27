#include<stdio.h>
#include <stdio.h>
#include <stdlib.h>
//definition of the structure
struct Node {
    int data;
    struct Node* next;//self refrential pointers
    struct Node* prev;
};
//creation of a newnode 
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    newNode->prev = NULL;
    return newNode;
}
//insertion function
void insertEnd(struct Node** head, int data) {
    struct Node* newNode = createNode(data);
    if (*head == NULL)//if the list is empty
     {
        *head = newNode;
        return;
    }
    struct Node* temp = *head;//a dummy pointer as head isnt supposed to change
    while (temp->next != NULL) {
        temp = temp->next;
    }
    temp->next = newNode;
    newNode->prev = temp;
}

void traverseForward(struct Node* head) {
    struct Node* temp = head;
    while (temp != NULL) {
        printf("%d ", temp->data);
        temp = temp->next;//updation
    }
    printf("\n");
}

void traverseReverse(struct Node* head) {
    if (head == NULL) return;
    struct Node* temp = head;
    while (temp->next != NULL) {
        temp = temp->next;//traverse to the end
    }
    while (temp != NULL) {
        printf("%d ", temp->data);
        temp = temp->prev;//travel to the back
    }
    printf("\n");
}

int main() {
    struct Node* head = NULL;
    insertEnd(&head, 10);
    insertEnd(&head, 20);
    insertEnd(&head, 30);
    insertEnd(&head, 40);
    traverseForward(head);
    traverseReverse(head);
    return 0;
}