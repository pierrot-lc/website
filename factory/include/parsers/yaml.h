#ifndef YAML_H
#define YAML_H

#include "tree.h"

/**
 * Get the first key node of the corresponding key content.
 */
Node *get_key(Node *, char *);

/**
 * Check if the key has a single value and if so returns its value. Otherwise
 * it stops the program immediately.
 */
char *get_value_scalar(Node *);

/**
 * Return the parsed tree of the given YAML source code.
 */
Node *parse_yaml(const char *);

#endif
