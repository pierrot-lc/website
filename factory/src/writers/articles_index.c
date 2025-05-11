#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>

#include "parsers/markdown.h"
#include "parsers/yaml.h"
#include "tree.h"
#include "ts_utils.h"

struct ArticleInfos {
  char path[256];
  char title[512];
};

static struct ArticleInfos get_article_infos(char *path) {
  char *source;

  struct ArticleInfos infos;
  Node *tree, *title;

  source = read_file(path);
  tree = parse_markdown(source);

  strlcpy(infos.path, path, 256);

  title = get_key(tree, "title");
  assert(title != NULL);
  strlcpy(infos.title, title->content, 512);

  free_tree(tree);
  return infos;
}

static void _article(FILE *file, struct ArticleInfos infos) {
  fprintf(file, "<li>\n");
  fprintf(file, "<h2><a href=\"%s\">%s</a></h2>\n", infos.path, infos.title);
  fprintf(file, "</li>\n");
}

void write_articles_index(FILE *file, const char *articles_dir) {
  char article_path[256];

  DIR *d;
  struct ArticleInfos infos;
  struct dirent *dir;

  d = opendir(articles_dir);

  if (d == NULL) {
    fprintf(stderr, "%s is not a directory", articles_dir);
    return;
  }

  fprintf(file, "<ul>\n");
  while ((dir = readdir(d)) != NULL) {
    if (dir->d_type != DT_REG)
      continue;

    strlcpy(article_path, articles_dir, 256);
    strlcat(article_path, "/", 256);
    strlcat(article_path, dir->d_name, 256);

    infos = get_article_infos(article_path);
    _article(file, infos);
  }
  fprintf(file, "</ul>\n");
}
