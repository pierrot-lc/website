#ifndef YAML_H
#define YAML_H

#include "tree.h"

/* Get the first key node of the corresponding key content. */
Node *get_key(Node *tree, char *key);
Node *parse_yaml(const char *source);

#endif
