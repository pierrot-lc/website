#ifndef TREE_H
#define TREE_H

typedef struct Node {
  unsigned int code;
  char *content;

  struct Node **children;
  unsigned int child_count;

  struct Node *parent;
} Node;

Node *create_node(unsigned int, char *);
void add_child(Node *parent, Node *child);
Node *tree_root(Node *);
void free_tree(Node *);
void print_tree(Node *);

#endif
