// Copyright (c) Acconeer AB, 2022-2026
// All rights reserved
// This file is subject to the terms and conditions defined in the file
// 'LICENSES/license_acconeer.txt', (BSD 3-Clause License) which is part
// of this source code package.

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "acc_definitions_a121.h"
#include "acc_definitions_common.h"
#include "acc_detector_presence.h"
#include "acc_hal_definitions_a121.h"
#include "acc_hal_integration_a121.h"
#include "acc_integration.h"
#include "acc_integration_log.h"
#include "acc_rss_a121.h"
#include "acc_sensor.h"

#include "acc_version.h"

/** \example example_detector_presence_with_iq_data_print.c
 * @brief This is an example on how the Detector Presence API can be used
 * @n
 * The example executes as follows:
 *   - Create a presence configuration
 *   - Create a sensor instance
 *   - Create a detector instance
 *   - Calibrate the sensor
 *   - Prepare the detector
 *   - Perform a sensor measurement and read out the data
 *   - Process the measurement and get detector result
 *   - Print the result
 *   - Destroy the configuration
 *   - Destroy the detector instance
 *   - Destroy the sensor instance
 */

typedef enum
{
	PRESENCE_PRESET_CONFIG_NONE = 0,
	PRESENCE_PRESET_CONFIG_SHORT_RANGE,
	PRESENCE_PRESET_CONFIG_MEDIUM_RANGE,
	PRESENCE_PRESET_CONFIG_LONG_RANGE,
	PRESENCE_PRESET_CONFIG_LOW_POWER_WAKEUP,
} presence_preset_config_t;

#define SENSOR_ID         (1U)
#define SENSOR_TIMEOUT_MS (2000U)

#define DEFAULT_PRESET_CONFIG PRESENCE_PRESET_CONFIG_MEDIUM_RANGE

static bool do_sensor_calibration(acc_sensor_t *sensor, acc_cal_result_t *cal_result, void *buffer, uint32_t buffer_size);

static void print_result(acc_detector_presence_result_t *result);

static void print_iq_data(acc_detector_presence_result_t *result, acc_detector_presence_metadata_t *metadata);

static void cleanup(acc_detector_presence_handle_t *presence_handle,
                    acc_detector_presence_config_t *presence_config,
                    acc_sensor_t                   *sensor,
                    void                           *buffer);

static void set_config(acc_detector_presence_config_t *presence_config, presence_preset_config_t preset);

int acc_example_detector_presence_with_iq_data_print(int argc, char *argv[]);

int acc_example_detector_presence_with_iq_data_print(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	acc_detector_presence_config_t  *presence_config = NULL;
	acc_detector_presence_handle_t  *presence_handle = NULL;
	acc_detector_presence_metadata_t metadata;
	acc_sensor_t                    *sensor      = NULL;
	void                            *buffer      = NULL;
	uint32_t                         buffer_size = 0U;

	printf("Acconeer software version %s\n", acc_version_get());

	const acc_hal_a121_t *hal = acc_hal_rss_integration_get_implementation();

	if (!acc_rss_hal_register(hal))
	{
		return EXIT_FAILURE;
	}

	presence_config = acc_detector_presence_config_create();
	if (presence_config == NULL)
	{
		printf("acc_detector_presence_config_create() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	set_config(presence_config, DEFAULT_PRESET_CONFIG);

	// Print the configuration
	acc_detector_presence_config_log(presence_config);

	presence_handle = acc_detector_presence_create(presence_config, &metadata);
	if (presence_handle == NULL)
	{
		printf("acc_detector_presence_create() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	if (!acc_detector_presence_get_buffer_size(presence_handle, &buffer_size))
	{
		printf("acc_detector_presence_get_buffer_size() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	buffer = acc_integration_mem_alloc(buffer_size);
	if (buffer == NULL)
	{
		printf("buffer allocation failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	acc_hal_integration_sensor_supply_on(SENSOR_ID);
	acc_hal_integration_sensor_enable(SENSOR_ID);

	sensor = acc_sensor_create(SENSOR_ID);
	if (sensor == NULL)
	{
		printf("acc_sensor_create() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	acc_cal_result_t cal_result;

	if (!do_sensor_calibration(sensor, &cal_result, buffer, buffer_size))
	{
		printf("do_sensor_calibration() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	if (!acc_detector_presence_prepare(presence_handle, presence_config, sensor, &cal_result, buffer, buffer_size))
	{
		printf("acc_detector_presence_prepare() failed\n");
		cleanup(presence_handle, presence_config, sensor, buffer);
		return EXIT_FAILURE;
	}

	while (true)
	{
		acc_detector_presence_result_t result;

		if (!acc_sensor_measure(sensor))
		{
			printf("acc_sensor_measure failed\n");
			cleanup(presence_handle, presence_config, sensor, buffer);
			return EXIT_FAILURE;
		}

		if (!acc_hal_integration_wait_for_sensor_interrupt(SENSOR_ID, SENSOR_TIMEOUT_MS))
		{
			printf("Sensor interrupt timeout\n");
			cleanup(presence_handle, presence_config, sensor, buffer);
			return EXIT_FAILURE;
		}

		if (!acc_sensor_read(sensor, buffer, buffer_size))
		{
			printf("acc_sensor_read failed\n");
			cleanup(presence_handle, presence_config, sensor, buffer);
			return EXIT_FAILURE;
		}

		if (!acc_detector_presence_process(presence_handle, buffer, &result))
		{
			printf("acc_detector_presence_process failed\n");
			cleanup(presence_handle, presence_config, sensor, buffer);
			return EXIT_FAILURE;
		}

		print_result(&result);
		print_iq_data(&result, &metadata);

		if (result.processing_result.data_saturated)
		{
			printf("Data saturated. The detector result is not reliable.\n");
		}

		if (result.processing_result.frame_delayed)
		{
			printf("Frame delayed. Could not read data fast enough.\n");
			printf("Try lowering the frame rate or call 'acc_sensor_read' more frequently.\n");
		}

		/* If "calibration_needed" is indicated, the sensor needs to be recalibrated. */
		if (result.processing_result.calibration_needed)
		{
			printf("Sensor recalibration needed ... \n");

			if (!do_sensor_calibration(sensor, &cal_result, buffer, buffer_size))
			{
				printf("do_sensor_calibration() failed\n");
				cleanup(presence_handle, presence_config, sensor, buffer);
				return EXIT_FAILURE;
			}

			printf("Sensor recalibration done!\n");

			/* Before measuring again, the sensor needs to be prepared through the detector. */
			if (!acc_detector_presence_prepare(presence_handle, presence_config, sensor, &cal_result, buffer, buffer_size))
			{
				printf("acc_detector_presence_prepare() failed\n");
				cleanup(presence_handle, presence_config, sensor, buffer);
				return EXIT_FAILURE;
			}
		}
	}

	cleanup(presence_handle, presence_config, sensor, buffer);

	printf("Application finished OK\n");

	return EXIT_SUCCESS;
}

static void cleanup(acc_detector_presence_handle_t *presence_handle,
                    acc_detector_presence_config_t *presence_config,
                    acc_sensor_t                   *sensor,
                    void                           *buffer)
{
	acc_hal_integration_sensor_disable(SENSOR_ID);
	acc_hal_integration_sensor_supply_off(SENSOR_ID);

	if (presence_config != NULL)
	{
		acc_detector_presence_config_destroy(presence_config);
	}

	if (presence_handle != NULL)
	{
		acc_detector_presence_destroy(presence_handle);
	}

	if (sensor != NULL)
	{
		acc_sensor_destroy(sensor);
	}

	if (buffer != NULL)
	{
		acc_integration_mem_free(buffer);
	}
}

static bool do_sensor_calibration(acc_sensor_t *sensor, acc_cal_result_t *cal_result, void *buffer, uint32_t buffer_size)
{
	bool           status              = false;
	bool           cal_complete        = false;
	const uint16_t calibration_retries = 1U;

	// Random disturbances may cause the calibration to fail. At failure, retry at least once.
	for (uint16_t i = 0; !status && (i <= calibration_retries); i++)
	{
		// Reset sensor before calibration by disabling/enabling it
		acc_hal_integration_sensor_disable(SENSOR_ID);
		acc_hal_integration_sensor_enable(SENSOR_ID);

		do
		{
			status = acc_sensor_calibrate(sensor, &cal_complete, cal_result, buffer, buffer_size);

			if (status && !cal_complete)
			{
				status = acc_hal_integration_wait_for_sensor_interrupt(SENSOR_ID, SENSOR_TIMEOUT_MS);
			}
		} while (status && !cal_complete);
	}

	if (status)
	{
		/* Reset sensor after calibration by disabling/enabling it */
		acc_hal_integration_sensor_disable(SENSOR_ID);
		acc_hal_integration_sensor_enable(SENSOR_ID);
	}

	return status;
}

static void print_iq_data(acc_detector_presence_result_t *result, acc_detector_presence_metadata_t *metadata)
{
	/* Get the iq data frame from the processing result */
	acc_int16_complex_t *frame = result->processing_result.frame;

	/* Get information about the frame from the sensor config */
	uint16_t sweeps_per_frame = acc_config_sweeps_per_frame_get(metadata->sensor_config);
	uint8_t  num_subsweeps    = acc_config_num_subsweeps_get(metadata->sensor_config);
	uint16_t sweep_length     = metadata->processing_metadata->frame_data_length / sweeps_per_frame;

	printf("BEGIN:Distance(m), [Frame], Amplitude\n");

	/* Loop over all subsweeps */
	for (uint8_t subsweep_idx = 0; subsweep_idx < num_subsweeps; subsweep_idx++)
	{
		/* Get information about the start point and step length from the sensor config */
		int32_t  start_point = acc_config_subsweep_start_point_get(metadata->sensor_config, subsweep_idx);
		uint16_t step_length = acc_config_subsweep_step_length_get(metadata->sensor_config, subsweep_idx);

		/* Get subsweep offset and length from the processing metadata */
		uint16_t subsweep_offset = metadata->processing_metadata->subsweep_data_offset[subsweep_idx];
		uint16_t subsweep_length = metadata->processing_metadata->subsweep_data_length[subsweep_idx];

		/* Loop over all points in subsweep */
		for (uint16_t point_idx = 0; point_idx < subsweep_length; point_idx++)
		{
			/* Print the point distance, in meters */
			float distance = acc_processing_points_to_meter(start_point + point_idx * step_length);
			printf("%" PRIfloat ", [", ACC_LOG_FLOAT_TO_INTEGER(distance));

			/* Perform a coherent mean calculation for the point over sweeps per frame */
			int32_t iq_point_real_acc = 0;
			int32_t iq_point_imag_acc = 0;

			/* Loop over all points in sweeps_per_frame */
			for (uint16_t sweep_idx = 0; sweep_idx < sweeps_per_frame; sweep_idx++)
			{
				uint16_t point_offset  = sweep_idx * sweep_length + subsweep_offset + point_idx;
				iq_point_real_acc     += frame[point_offset].real;
				iq_point_imag_acc     += frame[point_offset].imag;

				/* Make sure to print correct sign for imaginary part */
				char    sign = '+';
				int16_t imag = frame[point_offset].imag;

				if (imag < 0)
				{
					sign = '-';
					imag = -imag;
				}

				/* Print IQ point*/
				printf("%" PRIi16 "%c%" PRIi16 "i%s", frame[point_offset].real, sign, imag, ((sweep_idx + 1 == sweeps_per_frame) ? "" : ", "));
			}

			iq_point_real_acc = iq_point_real_acc / sweeps_per_frame;
			iq_point_imag_acc = iq_point_imag_acc / sweeps_per_frame;

			/* Calculate the absolute value of the IQ point */
			uint32_t iq_point_abs = (uint32_t)sqrt(iq_point_real_acc * iq_point_real_acc + iq_point_imag_acc * iq_point_imag_acc);

			/* Print the point absolute value */
			printf("], %" PRIu32 "\n", iq_point_abs);
		}
	}

	printf("END:Distance(m), [Frame], Amplitude\n");
}

static void print_result(acc_detector_presence_result_t *result)
{
	if (result->presence_detected)
	{
		printf("Motion\n");
	}
	else
	{
		printf("No motion\n");
	}

	// Score and distance are multiplied by 1000 to avoid printing floats
	printf("Intra presence score: %d, Inter presence score: %d, Distance (mm): %d\n",
	       (int)(result->intra_presence_score * 1000.0f),
	       (int)(result->inter_presence_score * 1000.0f),
	       (int)(result->presence_distance * 1000.0f));
}

static void set_config(acc_detector_presence_config_t *presence_config, presence_preset_config_t preset)
{
	switch (preset)
	{
		case PRESENCE_PRESET_CONFIG_NONE:
			// Add configuration of the detector here
			break;

		case PRESENCE_PRESET_CONFIG_SHORT_RANGE:
			acc_detector_presence_config_start_set(presence_config, 0.06f);
			acc_detector_presence_config_end_set(presence_config, 1.0f);
			acc_detector_presence_config_automatic_subsweeps_set(presence_config, true);
			acc_detector_presence_config_signal_quality_set(presence_config, 30.0f);
			acc_detector_presence_config_inter_frame_idle_state_set(presence_config, ACC_CONFIG_IDLE_STATE_DEEP_SLEEP);
			acc_detector_presence_config_sweeps_per_frame_set(presence_config, 16);
			acc_detector_presence_config_frame_rate_set(presence_config, 10.0f);
			acc_detector_presence_config_frame_rate_app_driven_set(presence_config, false);
			acc_detector_presence_config_reset_filters_on_prepare_set(presence_config, true);
			acc_detector_presence_config_intra_detection_set(presence_config, true);
			acc_detector_presence_config_intra_detection_threshold_set(presence_config, 1.4f);
			acc_detector_presence_config_intra_frame_time_const_set(presence_config, 0.15f);
			acc_detector_presence_config_intra_output_time_const_set(presence_config, 0.3f);
			acc_detector_presence_config_inter_detection_set(presence_config, true);
			acc_detector_presence_config_inter_detection_threshold_set(presence_config, 1.0f);
			acc_detector_presence_config_inter_frame_deviation_time_const_set(presence_config, 0.5f);
			acc_detector_presence_config_inter_frame_fast_cutoff_set(presence_config, 5.0f);
			acc_detector_presence_config_inter_frame_slow_cutoff_set(presence_config, 0.20f);
			acc_detector_presence_config_inter_output_time_const_set(presence_config, 2.0f);
			acc_detector_presence_config_inter_frame_presence_timeout_set(presence_config, 3);

			break;

		case PRESENCE_PRESET_CONFIG_MEDIUM_RANGE:
			acc_detector_presence_config_start_set(presence_config, 0.3f);
			acc_detector_presence_config_end_set(presence_config, 2.5f);
			acc_detector_presence_config_automatic_subsweeps_set(presence_config, true);
			acc_detector_presence_config_signal_quality_set(presence_config, 20.0f);
			acc_detector_presence_config_inter_frame_idle_state_set(presence_config, ACC_CONFIG_IDLE_STATE_DEEP_SLEEP);
			acc_detector_presence_config_sweeps_per_frame_set(presence_config, 16);
			acc_detector_presence_config_frame_rate_set(presence_config, 12.0f);
			acc_detector_presence_config_frame_rate_app_driven_set(presence_config, false);
			acc_detector_presence_config_reset_filters_on_prepare_set(presence_config, true);
			acc_detector_presence_config_intra_detection_set(presence_config, true);
			acc_detector_presence_config_intra_detection_threshold_set(presence_config, 1.3f);
			acc_detector_presence_config_intra_frame_time_const_set(presence_config, 0.15f);
			acc_detector_presence_config_intra_output_time_const_set(presence_config, 0.3f);
			acc_detector_presence_config_inter_detection_set(presence_config, true);
			acc_detector_presence_config_inter_detection_threshold_set(presence_config, 1.0f);
			acc_detector_presence_config_inter_frame_deviation_time_const_set(presence_config, 0.5f);
			acc_detector_presence_config_inter_frame_fast_cutoff_set(presence_config, 6.0f);
			acc_detector_presence_config_inter_frame_slow_cutoff_set(presence_config, 0.20f);
			acc_detector_presence_config_inter_output_time_const_set(presence_config, 2.0f);
			acc_detector_presence_config_inter_frame_presence_timeout_set(presence_config, 3);

			break;

		case PRESENCE_PRESET_CONFIG_LONG_RANGE:
			acc_detector_presence_config_start_set(presence_config, 5.0f);
			acc_detector_presence_config_end_set(presence_config, 7.5f);
			acc_detector_presence_config_automatic_subsweeps_set(presence_config, true);
			acc_detector_presence_config_signal_quality_set(presence_config, 10.0f);
			acc_detector_presence_config_inter_frame_idle_state_set(presence_config, ACC_CONFIG_IDLE_STATE_DEEP_SLEEP);
			acc_detector_presence_config_sweeps_per_frame_set(presence_config, 16);
			acc_detector_presence_config_frame_rate_set(presence_config, 12.0f);
			acc_detector_presence_config_frame_rate_app_driven_set(presence_config, false);
			acc_detector_presence_config_reset_filters_on_prepare_set(presence_config, true);
			acc_detector_presence_config_intra_detection_set(presence_config, true);
			acc_detector_presence_config_intra_detection_threshold_set(presence_config, 1.2f);
			acc_detector_presence_config_intra_frame_time_const_set(presence_config, 0.15f);
			acc_detector_presence_config_intra_output_time_const_set(presence_config, 0.3f);
			acc_detector_presence_config_inter_detection_set(presence_config, true);
			acc_detector_presence_config_inter_detection_threshold_set(presence_config, 0.8f);
			acc_detector_presence_config_inter_frame_deviation_time_const_set(presence_config, 0.5f);
			acc_detector_presence_config_inter_frame_fast_cutoff_set(presence_config, 6.0f);
			acc_detector_presence_config_inter_frame_slow_cutoff_set(presence_config, 0.20f);
			acc_detector_presence_config_inter_output_time_const_set(presence_config, 2.0f);
			acc_detector_presence_config_inter_frame_presence_timeout_set(presence_config, 3);

			break;

		case PRESENCE_PRESET_CONFIG_LOW_POWER_WAKEUP:
			acc_detector_presence_config_start_set(presence_config, 0.38f);
			acc_detector_presence_config_end_set(presence_config, 0.67f);
			acc_detector_presence_config_automatic_subsweeps_set(presence_config, false);
			acc_detector_presence_config_signal_quality_set(presence_config, 20.0f);
			acc_detector_presence_config_auto_step_length_set(presence_config, true);
			acc_detector_presence_config_auto_profile_set(presence_config, false);
			acc_detector_presence_config_profile_set(presence_config, ACC_CONFIG_PROFILE_5);
			acc_detector_presence_config_inter_frame_idle_state_set(presence_config, ACC_CONFIG_IDLE_STATE_DEEP_SLEEP);
			acc_detector_presence_config_hwaas_set(presence_config, 8);
			acc_detector_presence_config_sweeps_per_frame_set(presence_config, 8);
			acc_detector_presence_config_frame_rate_set(presence_config, 0.7f);
			acc_detector_presence_config_frame_rate_app_driven_set(presence_config, false);
			acc_detector_presence_config_reset_filters_on_prepare_set(presence_config, true);
			acc_detector_presence_config_intra_detection_set(presence_config, true);
			acc_detector_presence_config_intra_detection_threshold_set(presence_config, 1.7f);
			acc_detector_presence_config_intra_frame_time_const_set(presence_config, 0.3f);
			acc_detector_presence_config_intra_output_time_const_set(presence_config, 0.3f);
			acc_detector_presence_config_inter_detection_set(presence_config, true);
			acc_detector_presence_config_inter_detection_threshold_set(presence_config, 1.2f);
			acc_detector_presence_config_inter_frame_deviation_time_const_set(presence_config, 0.5f);
			acc_detector_presence_config_inter_frame_fast_cutoff_set(presence_config, 5.0f);
			acc_detector_presence_config_inter_frame_slow_cutoff_set(presence_config, 0.20f);
			acc_detector_presence_config_inter_output_time_const_set(presence_config, 0.5f);
			acc_detector_presence_config_inter_frame_presence_timeout_set(presence_config, 2);

			break;
	}
}
