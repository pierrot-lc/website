#ifndef TREE_H
#define TREE_H

typedef struct Column {
  char alignment[20];
  struct Node *cell;
} Column;

typedef struct Table {
  unsigned int ncols, nrows;
  struct Column **columns;
  struct Node ***cells;
} Table;

typedef struct Node {
  unsigned int code;

  union {
    char *content;
    struct Table *table;
  } data;

  struct Node **children;
  unsigned int child_count;
  struct Node *parent;
} Node;

Node *create_node(unsigned int, char *);
void add_child(Node *parent, Node *child);
Node *tree_root(Node *);
void free_tree(Node *);
void print_tree(Node *);
Node *search_node(Node *, unsigned int);

#endif
