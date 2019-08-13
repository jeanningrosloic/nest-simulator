/*
 *  recording_backend_ascii.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef RECORDING_BACKEND_ASCII_H
#define RECORDING_BACKEND_ASCII_H

// C++ includes:
#include <fstream>

#include "recording_backend.h"

/* BeginDocumentation

Write data to plain text files
##############################

The `ascii` recording backend writes collected data persistently to a
plain text ASCII file. It can be used for small to medium sized
simulations, where the ease of a simple data format outweights the
benefits of high-performance output operations.

This backend will open one file per recording device per thread on
each MPI process. This can entail a very high load on the file system
in large simulations. Especially on machines with distributed
filesystems using this backend can become prohibitively inefficient.
In case of experiencing such scaling problems, the :ref:`SIONlib
backend <sionlib_backend>` can be a possible alternative.

Filenames of data files are determined according to the following
pattern:

::

   data_path/data_prefix(label|model_name)-gid-vp.file_extension

The properties `data_path` and `data_prefix` are global kernel
properties. They can for example be set during repetitive simulation
protocols to separate the data originating from indivitual runs. The
`label` replaces the model name component if it is set to a non-empty
string. `gid` and `vp` denote the zero-padded global ID and virtual
process of the recorder writing the file. The filename ends in a dot
and the `file_extension`.

The life of a file starts with the call to ``Prepare`` and ends with
the call to ``Cleanup``. Data that is produced during successive calls
to ``Run`` inbetween a pair of ``Prepare`` and ``Cleanup`` calls will
be written to the same file, while the call to ``Run`` will flush all
data to the file, so it is available for immediate inspection.

In case, a file of the designated name for a new recording already
exists, the ``Prepare`` call will fail with a corresponding error
message. To instead overwrite the old file, the kernel property
`overwrite_files` can be set to *true* using ``SetKernelStatus``.  An
alternative way for avoiding name clashes is to re-set the kernel
properties `data_path` or `data_prefix`, so that another filename is
chosen.

Data format
+++++++++++

The first line written to any new file is an informational header
containing field names for the different data columns. The header
starts with a `#` character.

The first field of each record written is the global id of the neuron
the event originated from, i.e. the *source* of the event. This is
followed by the time of the measurement, the recorded floating point
values and the recorded integer values.

The format of the time field depends on the value of the property
`time_in_steps`. If set to *false* (which is the default), time is
written as a single floating point number representing the simulation
time in ms. If `time_in_steps` is *true*, the time of the event is
written as a pair of values consisting of the integer simulation time
step in units of the simulation resolution and the negative floating
point offset in ms from the next integer grid point.

.. note::
   The number of decimal places for all decimal numbers written can be
   controlled using the recorder property `precision`.

Parameter summary
+++++++++++++++++

`file_extension`
  A string (default: *"dat"*) that specifies the file name extension,
  without leading dot. The generic default was chosen, because the
  exact type of data cannot be known a priori.

`filenames`
  A list of the filenames where data is recorded to. This list has one
  entry per local thread and is a read-only property.

`label`
  A string (default: *""*) that replaces the model name component in
  the filename if it is set.

`precision`
  An integer (default: *3*) that controls the number of decimal places
  used to write decimal numbers to the output file.

`time_in_steps`
  A Boolean (default: *false*) specifying whether to write time in
  steps, i.e. in integer multiples of the simulation resolution plus a
  floating point number for the negative offset from the next grid
  point in ms, or just the simulation time in ms.

EndDocumentation */

namespace nest
{

/**
 * ASCII specialization of the RecordingBackend interface.
 *
 * RecordingBackendASCII maintains a data structure mapping one file
 * stream to every recording device instance on every thread. Files
 * are opened and inserted into the map during the enroll() call
 * (issued by the recorder's calibrate() function) and closed in
 * cleanup(), which is called on all registered recording backends by
 * IOManager::cleanup().
 */
class RecordingBackendASCII : public RecordingBackend
{
public:
  RecordingBackendASCII();

  ~RecordingBackendASCII() throw();

  void initialize() override;

  void finalize() override;

  void enroll( const RecordingDevice& device ) override;

  void disenroll( const RecordingDevice& device ) override;

  void set_value_names( const RecordingDevice& device,
    const std::vector< Name >& double_value_names,
    const std::vector< Name >& long_value_names ) override;

  void prepare() override;

  void cleanup() override;

  void pre_run_hook() override;

  /**
   * Flush files after a single call to Run
   */
  void post_run_hook() override;

  void write( const RecordingDevice&, const Event&, const std::vector< double >&, const std::vector< long >& ) override;

  void set_status( const DictionaryDatum& ) override;
  void get_status( DictionaryDatum& ) const override;

  void set_device_status( const RecordingDevice& device, const DictionaryDatum& d ) override;
  void get_device_status( const RecordingDevice& device, DictionaryDatum& ) const override;

private:
  /**
   * Build device file basename as being the device's label (or model
   * name if no label is given), the device's GID, and the virtual
   * process ID, all separated by dashes, followed by a dot and the
   * filename extension.
   */
  const std::string build_basename_( const RecordingDevice& device ) const;

  struct DeviceData
  {
    DeviceData() = delete;
    DeviceData( std::string );
    void set_value_names( const std::vector< Name >&, const std::vector< Name >& );
    void open_file();
    void write( const Event&, const std::vector< double >&, const std::vector< long >& );
    void flush_file();
    void close_file();
    void get_status( DictionaryDatum& ) const;
    void set_status( const DictionaryDatum& );

  private:
    long precision_;                         //!< Number of decimal places used when writing decimal values
    bool time_in_steps_;                     //!< Should time be recorded in steps (ms if false)
    std::string file_basename_;              //!< File name up to but not including the "."
    std::string file_extension_;             //!< File name extension without leading "."
    std::string filename_;                   //!< Full filename as determined and used by open_file()
    std::ofstream file_;                     //!< File stream to use for the device
    std::vector< Name > double_value_names_; //!< names for values of type double
    std::vector< Name > long_value_names_;   //!< names for values of type long
  };

  typedef std::vector< std::map< size_t, DeviceData > > data_map;
  data_map device_data_;
};

} // namespace

#endif // RECORDING_BACKEND_ASCII_H
