#include <assert.h>
#include <stdio.h>

#include "hash.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "writers/header.h"
#include "writers/utils.h"

static void nav(FILE *file, Node *tree) {
  Node *nav = get_key(tree, "nav");
  Node *home, *others, *href, *name;

  if (nav == NULL)
    return;

  assert((home = get_key(nav, "home")) != NULL);
  assert((others = get_key(nav, "others")) != NULL);

  fprintf(file, "<nav>\n");

  write_link(file, home->children[0]->content, "Home");
  fprintf(file, "\n");

  fprintf(file, "<ul>\n");
  for (int i = 0; i < others->child_count; i++) {
    assert(others->children[i]->code == HASH_BLOCK_SEQUENCE_ITEM);
    assert((href = get_key(others->children[i], "href")) != NULL);
    assert((name = get_key(others->children[i], "name")) != NULL);

    fprintf(file, "<li>");
    write_link(file, href->children[0]->content, name->children[0]->content);
    fprintf(file, "</li>\n");
  }
  fprintf(file, "</ul>\n");

  fprintf(file, "</nav>\n");
}

void write_header(FILE *file, Node *tree) {
  fprintf(file, "<header>\n");
  nav(file, tree);
  fprintf(file, "</header>\n");
}
