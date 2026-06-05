#include<stdio.h>
#include<stdlib.h>
struct node
{
    int data;
    struct node *next;
};
struct node *create(int data)
{
    struct node *newnode=(struct node *)malloc(sizeof(struct node));
    newnode->data=data;
    newnode->next=NULL;
    return newnode;
};
void insertb(struct node **head,int data)
{
    struct node *newnode=create(data);
    if(*head==NULL)
    {
        *head=newnode;
        return;
    }
    newnode->next=*head;
    *head=newnode;
}
void insertl(struct node **head,int data)
{
    struct node *newnode=create (data);
    if(*head==NULL)
    {
        *head=newnode;
        return;
    }
    struct node *temp=*head;
    while(temp->next !=NULL)
    {
        temp=temp->next;
    }
    temp->next=newnode;
    newnode->next=NULL;
}
void deleteb(struct node **head)
{
    if(*head==NULL)
    {
        return;
    }
    struct node *temp=*head;
    *head=temp->next;
    free(temp);
    return;
}
void deletel(struct node **head)
{
     if(*head==NULL)
    {
        return;
    }
    struct node *temp=*head;
    while((temp->next)->next !=0)
    {
        temp=temp->next;
    }
    free(temp->next);
    temp->next=NULL;

}
void insertp(struct node **head,int pos,int data)
{
    struct node *newnode=create(data);
    if(pos==1)
    {
        *head=newnode;
        return;
    }
    int i;
    struct node *temp=*head;
    for(i=1;i<pos-1;i++)
    {
        temp=temp->next;

    }
    if(temp==NULL)
    {
        printf("out\n");
    }
    newnode->next=temp->next;
    temp->next=newnode;


}
void print(struct node *head)
{
    if(head==NULL)
    {
        return;
    }
    struct node *temp=head;
    while(temp!=NULL)
    {
        printf("%d\t ",temp->data);
        temp=temp->next;

    }
    printf("\n");

}
int main()
{
    struct node *head=NULL;
    insertl(&head,20);
     insertl(&head,30);
      insertl(&head,40);
       insertl(&head,50);
        insertb(&head,60);
        print(head);
        deleteb(&head);
        deletel(&head);
        print(head);
        return 0;
}