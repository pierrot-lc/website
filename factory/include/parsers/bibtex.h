#ifndef BIBTEX_H
#define BIBTEX_H

#include "tree.h"

typedef struct Author {
  char firstname[50];
  char lastname[50];

  struct Author *next;
} Author;

/**
 * Return the destination node for the given bibtex entry. NULL if search has
 * failed.
 */
Node *search_bibliography_entry(Node *, char *);

/**
 * Return the first field that matches the given name in the bibtex entry.
 */
Node *get_field(Node *, char *);

/**
 * Parse the list of authors of a bibtex entry. This is highly not very
 * precise.
 * TODO: Find a better approach.
 */
Author *parse_authors(Node *);
void free_authors(Author *);

/**
 * Mark the corresponding entry as cited by adding a dedicated field in the
 * entry. If the entry is marked as cited, it will appear at the end of the
 * page.
 */
void mark_cited(Node *);

/**
 * Return the parsed tree of the given YAML source code.
 */
Node *parse_bibtex(const char *);

#endif
