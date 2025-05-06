#ifndef YAML_H
#define YAML_H

#include "tree.h"

/* Get the value node of the corresponding key node. */
Node *get_value(Node *, char *);
Node *parse_yaml(const char *);

#endif
