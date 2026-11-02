// Copyright (c) Acconeer AB, 2025
// All rights reserved
// This file is subject to the terms and conditions defined in the file
// 'LICENSES/license_acconeer.txt', (BSD 3-Clause License) which is part
// of this source code package.

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "acc_config.h"
#include "acc_hal_definitions_a121.h"
#include "acc_hal_integration_a121.h"
#include "acc_integration.h"
#include "acc_integration_log.h"
#include "acc_rss_a121.h"
#include "acc_sensor.h"
#include "acc_version.h"
#include "example_cargo.h"
#include "example_cargo_main.h"

#define SENSOR_ID         (1U)
#define SENSOR_TIMEOUT_MS (1000U)

/**
 * @brief Frees any allocated resources
 */
static void cleanup(acc_sensor_t *sensor, void *buffer, uint8_t *distance_cal_result_static, cargo_handle_t *handle);

/**
 * @brief Performs sensor calibration (with retry)
 */
static bool do_sensor_calibration(acc_sensor_t *sensor, void *buffer, uint32_t buffer_size, acc_cal_result_t *cal_result);

/**
 * @brief Performs cargo calibration (with retry)
 */
static bool do_cargo_calibration(acc_sensor_t                      *sensor,
                                 cargo_handle_t                    *cargo_handle,
                                 void                              *buffer,
                                 uint32_t                           buffer_size,
                                 acc_cal_result_t                  *sensor_cal_result,
                                 uint8_t                           *distance_cal_result_static,
                                 uint32_t                           distance_cal_result_static_size,
                                 acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic);

/**
 * @brief Print a processor result in a human-readable format
 */
static void print_cargo_result(const cargo_result_t *result, bool result_available);

int acc_example_cargo_main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	acc_sensor_t    *sensor            = NULL;
	acc_cal_result_t sensor_cal_result = {0};

	void    *buffer      = NULL;
	uint32_t buffer_size = 0;

	uint32_t                          distance_cal_result_static_size = 0U;
	uint8_t                          *distance_cal_result_static      = NULL;
	acc_detector_cal_result_dynamic_t distance_cal_result_dynamic     = {0};

	cargo_handle_t *cargo_handle = NULL;
	cargo_result_t  cargo_result = {0};

	cargo_config_t cargo_config = {0};

	printf("Acconeer software version %s\n", acc_version_get());

	const acc_hal_a121_t *hal = acc_hal_rss_integration_get_implementation();

	if (!acc_rss_hal_register(hal))
	{
		return EXIT_FAILURE;
	}

	cargo_config_initialize(&cargo_config, CARGO_PRESET_NO_LENS);

	cargo_config_log(&cargo_config);

	cargo_handle = cargo_handle_create(&cargo_config);
	if (cargo_handle == NULL)
	{
		printf("cargo_handle_create() failed\n");
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	acc_detector_presence_metadata_t presence_metadata = {0};
	if (cargo_presence_metadata_get(cargo_handle, &presence_metadata))
	{
		printf("Presence metadata\n");
		printf("presence_metadata.start_m:       %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(presence_metadata.start_m));
		printf("presence_metadata.end_m:         %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(presence_metadata.end_m));
		printf("presence_metadata.step_length_m: %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(presence_metadata.step_length_m));
		printf("presence_metadata.num_points:    %" PRIu16 "\n", presence_metadata.num_points);
	}
	else
	{
		printf("Cargo is not configured to have presence activated.\n");
	}

	if (!cargo_get_buffer_size(cargo_handle, &buffer_size))
	{
		printf("cargo_get_buffer_size() failed\n");
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	if (!cargo_get_distance_cal_result_static_size(cargo_handle, &distance_cal_result_static_size))
	{
		printf("cargo_get_distance_cal_result_static_size() failed\n");
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	buffer = acc_integration_mem_alloc(buffer_size);
	if (buffer == NULL)
	{
		printf("buffer allocation failed\n");
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	if (distance_cal_result_static_size > 0U)
	{
		distance_cal_result_static = acc_integration_mem_alloc(distance_cal_result_static_size);
		if (distance_cal_result_static == NULL)
		{
			printf("distance_cal_result_static buffer allocation failed\n");
			cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
			return EXIT_FAILURE;
		}
	}

	acc_hal_integration_sensor_supply_on(SENSOR_ID);
	acc_hal_integration_sensor_enable(SENSOR_ID);

	sensor = acc_sensor_create(SENSOR_ID);
	if (sensor == NULL)
	{
		printf("acc_sensor_create() failed\n");
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	if (!do_sensor_calibration(sensor, buffer, buffer_size, &sensor_cal_result))
	{
		printf("do_sensor_calibration() failed\n");
		acc_sensor_status(sensor);
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	// Reset sensor after calibration by disabling/enabling it
	acc_hal_integration_sensor_disable(SENSOR_ID);
	acc_hal_integration_sensor_enable(SENSOR_ID);

	if (!do_cargo_calibration(sensor,
	                          cargo_handle,
	                          buffer,
	                          buffer_size,
	                          &sensor_cal_result,
	                          distance_cal_result_static,
	                          distance_cal_result_static_size,
	                          &distance_cal_result_dynamic))
	{
		printf("do_cargo_calibration() failed\n");
		acc_sensor_status(sensor);
		cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
		return EXIT_FAILURE;
	}

	while (true)
	{
		if (cargo_config.activate_utilization_level)
		{
			if (!cargo_prepare_utilization(cargo_handle, sensor, &sensor_cal_result, buffer, buffer_size))
			{
				printf("cargo_prepare_utilization() failed\n");
				acc_sensor_status(sensor);
				cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
				return EXIT_FAILURE;
			}

			do
			{
				if (!acc_sensor_measure(sensor))
				{
					printf("acc_sensor_measure() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				if (!acc_hal_integration_wait_for_sensor_interrupt(SENSOR_ID, SENSOR_TIMEOUT_MS))
				{
					printf("Sensor interrupt timeout\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				if (!acc_sensor_read(sensor, buffer, buffer_size))
				{
					printf("acc_sensor_read() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				bool result_available = false;
				if (!cargo_process_utilization(cargo_handle,
				                               distance_cal_result_static,
				                               &distance_cal_result_dynamic,
				                               buffer,
				                               &cargo_result,
				                               &result_available,
				                               NULL))
				{
					printf("cargo_process_utilization() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				print_cargo_result(&cargo_result, result_available);

			} while (cargo_current_mode_get(cargo_handle) == CARGO_MODE_UTILIZATION);
		}
		if (cargo_config.activate_presence)
		{
			if (!cargo_prepare_presence(cargo_handle, sensor, &sensor_cal_result, buffer, buffer_size))
			{
				printf("cargo_prepare_presence() failed\n");
				acc_sensor_status(sensor);
				cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
				return EXIT_FAILURE;
			}

			do
			{
				if (!acc_sensor_measure(sensor))
				{
					printf("acc_sensor_measure() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				if (!acc_hal_integration_wait_for_sensor_interrupt(SENSOR_ID, SENSOR_TIMEOUT_MS))
				{
					printf("Sensor interrupt timeout\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				if (!acc_sensor_read(sensor, buffer, buffer_size))
				{
					printf("acc_sensor_read() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				if (!cargo_process_presence(cargo_handle, buffer, &cargo_result, NULL))
				{
					printf("cargo_process_presence() failed\n");
					acc_sensor_status(sensor);
					cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);
					return EXIT_FAILURE;
				}

				print_cargo_result(&cargo_result, true);

			} while (cargo_current_mode_get(cargo_handle) == CARGO_MODE_PRESENCE);
		}
	}

	cleanup(sensor, buffer, distance_cal_result_static, cargo_handle);

	printf("Application finished OK\n");

	return EXIT_SUCCESS;
}

static void cleanup(acc_sensor_t *sensor, void *buffer, uint8_t *distance_cal_result_static, cargo_handle_t *handle)
{
	acc_hal_integration_sensor_disable(SENSOR_ID);
	acc_hal_integration_sensor_supply_off(SENSOR_ID);

	if (sensor != NULL)
	{
		acc_sensor_destroy(sensor);
	}

	if (buffer != NULL)
	{
		acc_integration_mem_free(buffer);
	}

	if (distance_cal_result_static != NULL)
	{
		acc_integration_mem_free(distance_cal_result_static);
	}

	if (handle != NULL)
	{
		cargo_handle_destroy(handle);
	}
}

static bool do_sensor_calibration(acc_sensor_t *sensor, void *buffer, uint32_t buffer_size, acc_cal_result_t *cal_result)
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

	return status;
}

static bool do_cargo_calibration(acc_sensor_t                      *sensor,
                                 cargo_handle_t                    *cargo_handle,
                                 void                              *buffer,
                                 uint32_t                           buffer_size,
                                 acc_cal_result_t                  *sensor_cal_result,
                                 uint8_t                           *distance_cal_result_static,
                                 uint32_t                           distance_cal_result_static_size,
                                 acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic)
{
	bool status           = true;
	bool calibration_done = false;

	while (status && !calibration_done)
	{
		status = cargo_calibrate(cargo_handle,
		                         sensor,
		                         sensor_cal_result,
		                         buffer,
		                         buffer_size,
		                         distance_cal_result_static,
		                         distance_cal_result_static_size,
		                         distance_cal_result_dynamic,
		                         &calibration_done);

		if (status && !calibration_done)
		{
			status = acc_hal_integration_wait_for_sensor_interrupt(SENSOR_ID, SENSOR_TIMEOUT_MS);
		}
	}

	return status;
}

static void print_cargo_result(const cargo_result_t *result, bool result_available)
{
	static uint32_t frameno = 0U;
	frameno++;

	printf("frameno %" PRIu32 "\n", frameno);

	if (result_available)
	{
		if (result->utilization_valid)
		{
			printf("distance:             %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(result->distance));
			printf("level_m:              %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(result->level_m));
			printf("level_percent:        %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(result->level_percent));
		}

		if (result->presence_valid)
		{
			printf("presence_detected:    %s\n", result->presence_detected ? "true" : "false");
			printf("inter_presence_score: %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(result->inter_presence_score));
			printf("intra_presence_score: %" PRIfloat "\n", ACC_LOG_FLOAT_TO_INTEGER(result->intra_presence_score));
		}
	}
}
