/* NOT DONE - commented out */
#include "copyright.h"
/*==============================================================================
 * FILE: ath_log.c
 *
 * PURPOSE: Functions for controlling output to stderr and stdout.  On many
 *   machines, output to stderr and stdout does not scale well (to 10^{4-5}
 *   processors).  However, having output from all MPI processes can be useful
 *   for debugging.  Thus, some kind of runtime control that allows output to
 *   stderr and stdout be turned off for production runs, or redirected to
 *   files for debugging, is very helpful.
 *
 * CONTAINS PUBLIC FUNCTIONS:
 *   ath_log_name_init() - initializes log filenames
 *   ath_log_out_open()  - opens out log file
 *   ath_log_err_open()  - opens err log file
 *   ath_log_set_level() - sets logging level from input arguments
 *   ath_log_open()      - calls output/error log file open if lazy == 0
 *   ath_log_close()     - closes output/error log file close
 *   athout_fp()         - returns pointer to out logfile
 *   atherr_fp()         - returns pointer to err logfile
 *   ath_flush_out()     - flushes out log file
 *   ath_flush_err()     - flushes err log file
 *   ath_perr()          - writes to err file
 *   ath_pout()          - writes to out file
 *============================================================================*/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include "prototypes.h"

/* output and error log levels */
static int out_level = 0, err_level = 0;

/* Simulation output and error file pointers */
static FILE *ath_fp_out = NULL, *ath_fp_err = NULL;

/* Simulation output and error log file names */
static char *out_fname = NULL, *err_fname = NULL;

/* Logical flags indicating that the files need to be opened */
static int open_out_flag = 0, open_err_flag = 0;

/*----------------------------------------------------------------------------*/
/* ath_log_name_init: creates log file name
 */

static void ath_log_name_init(const char *basename)
{
/* ath_fp_out filename */
/*
  if((out_fname = fname_construct(basename, 0, 0, NULL, "out")) == NULL){
    fprintf(stderr,"[ath_log_open]: Error constructing filename \"%s.out\"\n",
	    basename);
    exit(EXIT_FAILURE);
*/
//  }

/* ath_fp_err filename */
/*
  if((err_fname = fname_construct(basename, 0, 0, NULL, "err")) == NULL){
    fprintf(stderr,"[ath_log_open]: Error constructing filename \"%s.err\"\n",
	    basename);
    exit(EXIT_FAILURE);
  }

  return;
*/
}

/*----------------------------------------------------------------------------*/
/* ath_log_out_open: opens output log file (containing output to stdout)
 */

static int ath_log_out_open(void)
{
  open_out_flag = 0; /* zero the flag to open the file, only 1 try if we fail */

/* open the ath_fp_out file pointer */
  if((ath_fp_out = fopen(out_fname, "w")) == NULL){
    fprintf(stderr,"[ath_log_open]: Failed to open ath_fp_out as \"%s\"\n",
	    out_fname);
    return 1;
  }

  return 0;
}

/*----------------------------------------------------------------------------*/
/* ath_log_err_open: opens error log file (containing output to stderr)
 */

static int ath_log_err_open(void)
{
  open_err_flag = 0; /* zero the flag to open the file, only 1 try if we fail */

/* open the ath_fp_err file pointer */
  if((ath_fp_err = fopen(err_fname, "w")) == NULL){
    fprintf(stderr,"[ath_log_open]: Failed to open ath_fp_err as \"%s\"\n",
	    err_fname);
    return 1;
  }

  return 0;
}

/*----------------------------------------------------------------------------*/
/* ath_log_set_level: sets "output level" and "error level" from input
 *   arguments, global parameters used by many functions in this file.
 *   Called from main(), where the parameters are parsed from athinput.
 */

void ath_log_set_level(const int out, const int err)
{
  out_level = out;
  err_level = err;

  return;
}

/*----------------------------------------------------------------------------*/
/* ath_log_open: opens output and error log files if "lazy" flag is false (0).
 *   If "lazy" is true, then the ath_log_open() call does not actually open the
 *   files, but rather prepares to open them on the first invocation of either
 *   ath_pout() ath_perr(), athout_fp() or atherr_fp().  This is useful for
 *   parallel jobs where the children, if made sufficiently quiet would
 *   otherwise generate a large number of empty files. -- T. A. Gardiner -- 
 */

void ath_log_open(const char *basename, const int lazy)
{
  ath_log_name_init(basename); /* Initialize the log file names */

/* Indicate that these files should be opened before any action is done. */
  open_out_flag = open_err_flag = 1;

  if(lazy == 0){
    /* open the files now and exit on failure! */
    if(ath_log_out_open() || ath_log_err_open())
      exit(EXIT_FAILURE);
  }

  return;
}

/*----------------------------------------------------------------------------*/
/* ath_log_close: closes output and error log files 
 */

void ath_log_close(void)
{
/* clear the flags to open the log files */
  open_out_flag = open_err_flag = 0;

/* free the log file names */
  if(out_fname != NULL){ free(out_fname);  out_fname = NULL; }
  if(err_fname != NULL){ free(err_fname);  err_fname = NULL; }

/* close the files */
  if(ath_fp_out != NULL){ fclose(ath_fp_out);  ath_fp_out = NULL; }
  if(ath_fp_err != NULL){ fclose(ath_fp_err);  ath_fp_err = NULL; }

  return;
}

/*----------------------------------------------------------------------------*/
/* athout_fp: open output file if needed and return a pointer to file
 */

FILE *athout_fp(void)
{
/* Open the output log file if it needs to be opened */
  if(open_out_flag) ath_log_out_open();

/* Return either the output log file pointer or stdout */
  return (ath_fp_out == NULL ? stdout : ath_fp_out);
}

/*----------------------------------------------------------------------------*/
/* atherr_fp: open error file if needed and return a pointer to file
 */

FILE *atherr_fp(void)
{
/* Open the error log file if it needs to be opened */
  if(open_err_flag) ath_log_err_open();

/* Return either the error log file pointer or stderr */
  return (ath_fp_err == NULL ? stderr : ath_fp_err);
}

/*----------------------------------------------------------------------------*/
/* ath_flush_out: flush output file buffer
 */

void ath_flush_out(void)
{
  FILE *fp= (ath_fp_out == NULL ? stdout : ath_fp_out);
  fflush(fp);
}

/*----------------------------------------------------------------------------*/
/* ath_flush_err: flush error file buffer
 */

void ath_flush_err(void)
{
  FILE *fp= (ath_fp_err == NULL ? stderr : ath_fp_err);
  fflush(fp);
}

/*----------------------------------------------------------------------------*/
/* ath_perr: output variable argument string to "error log file".   Should
 *   be used in place of printf(stderr,...)
 */

int ath_perr(const int level, const char *fmt, ...)
{
  va_list ap;
  int iret = 0;
  FILE *fp;

  if(level <= err_level){
    /* Open the error log file if it needs to be opened */
    if(open_err_flag) ath_log_err_open();

    /* Use either the error log file pointer or stderr */
    fp = (ath_fp_err == NULL ? stderr : ath_fp_err);

    va_start(ap, fmt);            /* ap starts after the fmt parameter */
    iret = vfprintf(fp, fmt, ap); /* print the error message to stderr */
    va_end(ap);                   /* end stdargs (clean up the va_list ap) */
  }

  return iret;
}

/*----------------------------------------------------------------------------*/
/* ath_pout: output variable argument string to "output log file".   Should
 *   be used in place of printf(stdout,...)
 */

int ath_pout(const int level, const char *fmt, ...)
{
  va_list ap;
  int iret = 0;
  FILE *fp;

  if(level <= out_level){
    /* Open the output log file if it needs to be opened */
    if(open_out_flag) ath_log_out_open();

    /* Use either the output log file pointer or stdout */
    fp = (ath_fp_out == NULL ? stdout : ath_fp_out);

    va_start(ap, fmt);            /* ap starts after the fmt parameter */
    iret = vfprintf(fp, fmt, ap); /* print the error message to stderr */
    va_end(ap);                   /* end stdargs (clean up the va_list ap) */
  }

  return iret;
}
