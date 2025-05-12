#include <stdio.h>

#include "tree.h"
#include "writers/articles_index.h"
#include "writers/body_main.h"
#include "writers/head.h"
#include "writers/header.h"

void articles_index_page(FILE *file, const char *articles_dir, Node *config) {
  fprintf(file, "<!DOCTYPE html>\n");
  fprintf(file, "<html>\n\n");

  write_head(file, config);

  fprintf(file, "\n<body>\n");
  write_articles_index(file, articles_dir);
  fprintf(file, "</body>\n\n");

  fprintf(file, "</html>\n\n");
}

void write_html(FILE *file, Node *tree) {
  fprintf(file, "<!DOCTYPE html>\n");
  fprintf(file, "<html>\n\n");

  write_head(file, tree);

  write_header(file, tree);
  fprintf(file, "\n<body>\n");
  write_body_main(file, tree);
  fprintf(file, "</body>\n\n");

  fprintf(file, "</html>\n");
}
