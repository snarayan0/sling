#ifndef SLING_TASK_DASHBOARD_H_
#define SLING_TASK_DASHBOARD_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sling/base/types.h"
#include "sling/http/http-server.h"
#include "sling/task/job.h"
#include "sling/util/mutex.h"

namespace sling {
namespace task {

// Dashboard for monitoring jobs.
class Dashboard : public Monitor {
 public:
  // List of counters.
  typedef std::vector<std::pair<string, int64>> CounterList;

  ~Dashboard();

  // Register job status service.
  void Register(HTTPServer *http);

  // Handle job status queries.
  void HandleStatus(HTTPRequest *request, HTTPResponse *response);

  // Job monitor interface.
  void OnJobStart(Job *job) override;
  void OnJobDone(Job *job) override;

 private:
  // Status for active or completed job.
  struct JobStatus {
    JobStatus(Job *job) : job(job) {}
    Job *job;               // job object or null if job has completed
    string name;            // job name
    int64 started = 0;      // job start time
    int64 ended = 0;        // job completion time
    CounterList counters;   // final job counters
  };

  // List of active and completed jobs.
  std::vector<JobStatus *> jobs_;

  // Map of active jobs.
  std::unordered_map<Job *, JobStatus *> active_jobs_;

  // Mutex for serializing access to dashboard.
  Mutex mu_;
};

}  // namespace task
}  // namespace sling

#endif  // SLING_TASK_DASHBOARD_H_

