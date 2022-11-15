#include <mpi.h>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            exit( mpi_status );                                                       \
        }                                                                             \
    }

inline void initialize_mpi(int &rank, int &size) {
  MPI_Comm allcomm;
  char verstring[MPI_MAX_LIBRARY_VERSION_STRING];
  char nodename[MPI_MAX_PROCESSOR_NAME];
  int version, subversion, verstringlen, nodestringlen;
  allcomm = MPI_COMM_WORLD;
  MPI_Init(NULL, NULL);
  MPI_Comm_size(allcomm, &size);
  MPI_Comm_rank(allcomm, &rank);
  MPI_Get_processor_name(nodename, &nodestringlen);
  MPI_Get_version(&version, &subversion);
  MPI_Get_library_version(verstring, &verstringlen);
  if (rank == 0) {
    printf("Version %d, subversion %d\n", version, subversion);
    printf("Library <%s>\n", verstring);
  }
}
