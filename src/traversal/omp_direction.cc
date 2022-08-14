// Copyright 2022 MIT
// Authors: Xuhao Chen <cxh@mit.edu>
#include "graph.h"
#include "bitmap.h"
#include "sliding_queue.h"
#include "platform_atomics.h"

//int64_t BUStep(Graph &g, VertexList &parent, Bitmap &front, Bitmap &next) {
int64_t BUStep(Graph &g, VertexList &depths, Bitmap &front, Bitmap &next) {
  int64_t awake_count = 0;
  next.reset();
  #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
  for (vidType dst = 0; dst < g.V(); dst ++) {
    //if (parent[dst] < 0) {
    if (depths[dst] < 0) { // not visited
      for (auto src : g.in_neigh(dst)) {
        if (front.get_bit(src)) {
          //parent[dst] = src;
          depths[dst] = depths[src] + 1;
          awake_count++;
          next.set_bit(dst);
          break;
        }
      }
    }
  }
  return awake_count;
}

//int64_t TDStep(Graph &g, VertexList &parent, SlidingQueue<vidType> &queue) {
int64_t TDStep(Graph &g, VertexList &depths, SlidingQueue<vidType> &queue) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for reduction(+ : scout_count)
    for (auto *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      auto src = *q_iter;
      for (auto dst : g.out_neigh(src)) {
        //auto curr_val = parent[dst];
        auto curr_val = depths[dst];
        if (curr_val < 0) { // not visited
          //if (compare_and_swap(parent[dst], curr_val, src)) {
          if (compare_and_swap(depths[dst], curr_val, depths[src] + 1)) {
            lqueue.push_back(dst);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}

void QueueToBitmap(const SlidingQueue<vidType> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto *q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    auto u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(vidType m, const Bitmap &bm, SlidingQueue<vidType> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<vidType> lqueue(queue);
    #pragma omp for
    for (vidType n = 0; n < m; n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

VertexList InitParent(vidType m, vidType *degrees) {
  VertexList parent(m);
  #pragma omp parallel for
  for (vidType n = 0; n < m; n++)
    parent[n] = degrees[n] != 0 ? -degrees[n] : -1;
  return parent;
}

VertexList InitDepth(vidType m, vidType *degrees) {
  VertexList depths(m);
  #pragma omp parallel for
  for (vidType n = 0; n < m; n++)
    depths[n] = degrees[n] != 0 ? -degrees[n] : -1;
  return depths;
}

void BFSSolver(Graph &g, vidType source, vidType *dist) {
  if (!g.has_reverse_graph()) {
    std::cout << "This algorithm requires the reverse graph constructed for directed graph\n";
    std::cout << "Please set reverse to 1 in the command line\n";
    exit(1);
  }
  auto m = g.V();
  int num_threads = 1;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP Breadth-first Search (" << num_threads << "threads\n";
  int alpha = 15, beta = 18;
  std::vector<vidType> degrees(m, 0);
  #pragma omp parallel for
  for (vidType i = 0; i < m; i ++) {
    degrees[i] = g.get_degree(i);
  }
  //VertexList parent = InitParent(m, degrees);
  //parent[source] = source;
  //VertexList depths(m, MYINFINITY);
  auto depths = InitDepth(m, &degrees[0]);
  depths[source] = 0;
  SlidingQueue<vidType> queue(m);
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(m);
  curr.reset();
  Bitmap front(m);
  front.reset();
  int64_t edges_to_check = g.E();
  int64_t scout_count = degrees[source];
  int iter = 0;
  Timer t;
  t.Start();
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      QueueToBitmap(queue, front);
      awake_count = queue.size();
      queue.slide_window();
      do {
        ++ iter;
        old_awake_count = awake_count;
        awake_count = BUStep(g, depths, front, curr);
        front.swap(curr);
        //printf("BU: ");
        //printf("iteration=%d, num_frontier=%ld\n", iter, awake_count);
      } while ((awake_count >= old_awake_count) ||
          (awake_count > m / beta));
      BitmapToQueue(m, front, queue);
      scout_count = 1;
    } else {
      ++ iter;
      edges_to_check -= scout_count;
      scout_count = TDStep(g, depths, queue);
      queue.slide_window();
      //printf("TD: (scout_count=%ld) ", scout_count);
      //printf("TD: iteration=%d, num_frontier=%ld\n", iter, queue.size());
    }
  }
  t.Stop();

  printf("\titerations = %d.\n", iter);
  printf("\truntime [omp_direction] = %f ms.\n", t.Millisecs());
  #pragma omp parallel for
  for (vidType i = 0; i < m; i ++) {
    if (depths[i]>=0) dist[i] = depths[i]; 
    else dist[i] = MYINFINITY;
  }
  return;
}

void SSSPSolver(Graph &g, vidType source, elabel_t *dist, int delta) {}
