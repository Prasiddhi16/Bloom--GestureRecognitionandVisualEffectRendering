#include<stdio.h>
#include<stdlib.h>
struct Node
{
    int data;
    struct Node *left;
    struct Node *right;

};
struct Node *create(int data)
{
    struct Node *root=(struct Node *)malloc(sizeof(struct Node));
    root->data=data;
    root->left=NULL;
    root->right=NULL;
    return root;
};
struct Node * insert(struct Node *root,int data)
{
    if(root==NULL)
    {
        return create(data);
    }
    if(data<root->data)
    {
        root->left=insert(root->left,data);
    }
    if(data>root->data)
    {
        root->right=insert(root->right,data);
    }
    return root;
};
void inorder(struct Node *root)
{
    if(root==NULL)
    {
        return;
    }
    inorder(root->left);
    printf("%d\t",root->data);
    inorder(root->right);
 
}
void preorder(struct Node *root)
{
    if(root==NULL)
    {
        return;
    }
    printf("%d\t",root->data);
     preorder(root->left);
    preorder(root->right);
 
}
void postorder(struct Node *root)
{
    if(root==NULL)
    {
        return;
    }
    postorder(root->left);
     postorder(root->right);
    printf("%d\t",root->data);
  
   
}
int main()
{
    struct Node *root=NULL;
    root=insert(root,50);
    root=insert(root,20);
    root=insert(root,60);
    root=insert(root,30);
    root=insert(root,10);
    inorder(root);
     printf("\n");
    preorder(root);
     printf("\n");
    postorder(root);
    return 0;
}