import sys
import traceback
from enum import Enum

from microcircuit.microcircuit_run import run_colun
#from microcircuit.synfire_if_curr_exp import run_chain
from spinn_front_end_common.utilities import globals_variables


class AggregatorRun(object):

    # the number of runs to aggregate over
    N_RUNS = 10
    #N_RUNS = 1

    # the number of times to retry without crashing fully
    N_RETRIES = 3

    # the cfg params basic needs
    BASIC_DATA = (
        "[Buffers]\n\n" + "use_auto_pause_and_resume = True\n\n" +
        "[Simulation]\n\n" + "incoming_spike_buffer_size = 512\n\n" +
        "[Machine] \n\ntimeScaleFactor = 600\n\n")

    # the cfg params for only using none expander
    NO_EXPANDER = "[Synapses]\n\nuse_expander = False\n\n"

    # the cfg params for using the expander
    EXPANDER = "[Synapses]\n\nuse_expander = True\n\n"

    # the cfg params for python only
    USE_PYTHON = "[Java] \n\nuse_java = False\n\n"

    # the cfg params for java only
    USE_JAVA = (
        "[Java] \n\nuse_java = True\n\n" +
        "spinnaker.parallel_tasks = 1\n\n")

    USE_JAVA_PARALLEL = (
        "[Java] \n\nuse_java = True\n\n" +
        "spinnaker.parallel_tasks = 4\n\n")

    # the cfg params for using speed up stuff (in [Machine])
    USE_SPEED_UP = (
            "enable_advanced_monitor_support = True\n\n" +
            "enable_reinjection = True\n\n" +
            "disable_advanced_monitor_usage_for_data_in = False\n\n")

    # the cfg params for using sdp (in [Machine])
    NO_SPEED_UP = (
        "enable_advanced_monitor_support = False\n\n" +
        "enable_reinjection = False\n\n" +
        "disable_advanced_monitor_usage_for_data_in = False\n\n")

    USE_MPIF = (
        "[SpinnMan] \n\n" +
        "multi_packets_in_flight_n_channels = 8\n\n" +
        "multi_packets_in_flight_channel_waits = 7\n\n"
    )

    # the cfg params for using MPIF (in [Machine])
    USE_SDP = (
            "[SpinnMan] \n\n" +
            "multi_packets_in_flight_n_channels = 1\n\n" +
            "multi_packets_in_flight_channel_waits = 0\n\n")

    class STATES(Enum):
        """Regions for populations."""
        USE_SDP = 0
        USE_MPIF = 1
        USE_PROTOCOL_PYTHON = 2
        USE_PROTOCOL_JAVA = 3
        USE_PROTOCOL_JAVA_PARALLEL = 4
        USE_PROTOCOL_JAVA_EXPANDER = 5
        USE_PROTOCOL_JAVA_EXPANDER_PARALLEL = 6
        USE_PROTOCOL_PYTHON_EXPANDER = 7

    # filepath name
    CFG_FILE_NAME = "spynnaker.cfg"
    OUT_PUT_FILE = "results.txt"
    FAIL_PATH = "failed.txt"
    SIZE_TOTAL_PATH = "size_totals.txt"
    MATRIX_TOTALS_PATH = "matrix_totals.txt"
    EXPANDER_TOTALS_PATH = "expander_totals.txt"

    def __init__(self):
        pass

    def _set_config_python_sdp(self):
        """ sets the spynnaker.cfg so that it will run sdp in python

        :return:
        """
        output = open(self.CFG_FILE_NAME, "w")

        output.write(self.BASIC_DATA)
        output.write(self.NO_SPEED_UP)
        output.write(self.USE_PYTHON)
        output.write(self.USE_SDP)
        output.write(self.NO_EXPANDER)
        output.flush()
        output.close()

    def _set_python_mpif(self):
        """ sets the spynnaker.cfg so that it will run mpif in python

        :return:
        """
        output = open(self.CFG_FILE_NAME, "w")

        output.write(self.BASIC_DATA)
        output.write(self.NO_SPEED_UP)
        output.write(self.USE_PYTHON)
        output.write(self.USE_MPIF)
        output.write(self.NO_EXPANDER)
        output.flush()
        output.close()

    def _set_config_java_expander(self):
        """ sets the spynnaker.cfg so that it will run java in no parallel

        :return:
        """
        output = open(self.CFG_FILE_NAME, "w")

        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_JAVA)
        output.write(self.USE_MPIF)
        output.write(self.EXPANDER)
        output.flush()
        output.close()

    def _set_config_java_no_expander(self):
        """ sets the spynnaker.cfg so that it will run java in no parallel

        :return:
        """
        output = open(self.CFG_FILE_NAME, "w")

        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_JAVA)
        output.write(self.USE_MPIF)
        output.write(self.NO_EXPANDER)
        output.flush()
        output.close()

    def _set_config_java_parallel_no_expander(self):
        """ sets the spynnaker.cfg so that it will run

                :return:
                """
        output = open(self.CFG_FILE_NAME, "w")
        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_JAVA_PARALLEL)
        output.write(self.USE_MPIF)
        output.write(self.NO_EXPANDER)
        output.flush()
        output.close()

    def _set_config_java_parallel_expander(self):
        """ sets the spynnaker.cfg so that it will run

                :return:
                """
        output = open(self.CFG_FILE_NAME, "w")
        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_JAVA_PARALLEL)
        output.write(self.USE_MPIF)
        output.write(self.EXPANDER)
        output.flush()
        output.close()

    def _set_python_protocol(self):
        output = open(self.CFG_FILE_NAME, "w")
        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_PYTHON)
        output.write(self.USE_MPIF)
        output.write(self.NO_EXPANDER)
        output.flush()
        output.close()

    def _set_python_expander_protocol(self):
        output = open(self.CFG_FILE_NAME, "w")
        output.write(self.BASIC_DATA)
        output.write(self.USE_SPEED_UP)
        output.write(self.USE_PYTHON)
        output.write(self.USE_MPIF)
        output.write(self.EXPANDER)
        output.flush()
        output.close()

    def _protected_run(self, state, iteration):
        passed = False
        failed = False
        attempt = 0
        total_sdram = None
        matrix = None
        expander = None
        data_extraction_time = None
        data_loading_time_dsg = None
        data_loading_time_dse = None
        data_loading_time_expand = None
        data_extraction_size = None
        io_time = None

        # run
        while not passed and not failed:
            try:
                print("running {}:{}".format(state, iteration))
                (total_sdram, matrix, expander, data_extraction_time,
                 data_loading_time_dsg, data_loading_time_dse,
                 data_loading_time_expand, data_extraction_size, io_time) = \
                    run_colun()
                #(total_sdram, matrix, expander, data_extraction_time,
                # data_loading_time_dsg, data_loading_time_dse,
                 #data_loading_time_expand, data_extraction_size, io_time) = \
                #   run_chain()
                passed = True
            except Exception as e:
                attempt += 1
                fail = open(self.FAIL_PATH, "a")
                ex_type, ex_value, ex_traceback = sys.exc_info()
                trace_back = traceback.extract_tb(ex_traceback)

                fail.write(
                    "failed for state {}:{} retry {} with error {} "
                    "trace {}\n".format(
                        state, iteration, attempt, str(e), trace_back))
                if attempt >= self.N_RETRIES:
                    failed = True
                    fail.write("failed fully for state {}:{}\n".format(
                        state, iteration))
                fail.flush()
                fail.close()
                globals_variables.unset_simulator()

        # get data
        if passed:
            out = open(self.OUT_PUT_FILE, "a")
            out.write(
                "[{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}]"
                "\n".format(
                    state, iteration, total_sdram[(-1, -1, -1, -1)],
                    matrix[(-1, -1, -1)], expander[(-1, -1, -1)],
                    data_extraction_time, data_loading_time_dsg,
                    data_loading_time_dse, data_loading_time_expand,
                    data_extraction_size, io_time))
            out.flush()
            out.close()

            # size stores
            tots = open(self.SIZE_TOTAL_PATH, "a")
            tots.write("{}\n".format(total_sdram))
            tots.flush()
            tots.close()

            matrixs_tots = open(self.MATRIX_TOTALS_PATH, "a")
            matrixs_tots.write("{}\n".format(matrix))
            matrixs_tots.flush()
            matrixs_tots.close()

            expander_tots = open(self.EXPANDER_TOTALS_PATH, "a")
            expander_tots.write("{}\n".format(expander))
            expander_tots.flush()
            expander_tots.close()

    def __call__(self):

        # clear the old files so that no mixings
        x = open(self.FAIL_PATH, "w")
        x.close()

        y = open(self.OUT_PUT_FILE, "w")
        y.close()

        z = open(self.SIZE_TOTAL_PATH, "w")
        z.close()

        x = open(self.MATRIX_TOTALS_PATH, "w")
        x.close()

        y = open(self.EXPANDER_TOTALS_PATH, "w")
        y.close()

        #self._set_config_java_parallel_expander()
        #for run_id in range(0, self.N_RUNS):
        #    self._protected_run(
        #        self.STATES.USE_PROTOCOL_JAVA_EXPANDER_PARALLEL, run_id)

        #self._set_python_expander_protocol()
        #for run_id in range(0, self.N_RUNS):
        #    self._protected_run(
        #        self.STATES.USE_PROTOCOL_PYTHON_EXPANDER, run_id)

        self._set_config_java_expander()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(
                self.STATES.USE_PROTOCOL_JAVA_EXPANDER, run_id)

        self._set_python_protocol()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(self.STATES.USE_PROTOCOL_PYTHON, run_id)

        self._set_python_expander_protocol()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(
                self.STATES.USE_PROTOCOL_PYTHON_EXPANDER, run_id)

        #self._set_config_java_expander()
        #for run_id in range(0, self.N_RUNS):
        #    self._protected_run(
        #        self.STATES.USE_PROTOCOL_JAVA_EXPANDER, run_id)

        self._set_config_python_sdp()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(self.STATES.USE_SDP, run_id)

        self._set_python_mpif()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(self.STATES.USE_MPIF, run_id)

        self._set_config_java_no_expander()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(self.STATES.USE_PROTOCOL_JAVA, run_id)

        self._set_config_java_parallel_no_expander()
        for run_id in range(0, self.N_RUNS):
            self._protected_run(
                self.STATES.USE_PROTOCOL_JAVA_PARALLEL, run_id)

        print("completed")


if __name__ == '__main__':
    x = AggregatorRun()
    x()
