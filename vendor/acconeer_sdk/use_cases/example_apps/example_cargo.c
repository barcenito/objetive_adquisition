// Copyright (c) Acconeer AB, 2025
// All rights reserved
// This file is subject to the terms and conditions defined in the file
// 'LICENSES/license_acconeer.txt', (BSD 3-Clause License) which is part
// of this source code package.

#include <stdbool.h>
#include <stddef.h>

#include "acc_config.h"
#include "acc_detector_distance.h"
#include "acc_detector_presence.h"
#include "acc_integration.h"
#include "acc_integration_log.h"
#include "acc_sensor.h"
#include "example_cargo.h"

#define MODULE                              "example_cargo"
#define CARGO_CONFIG_MAGIC                  0xACC0CA60
#define CARGO_PRESENCE_RUN_TIME_S           5.0f
#define CARGO_PRESENCE_MIN_UPDATE_RATE      1.0f
#define CARGO_PRESENCE_MIN_SWEEPS_PER_FRAME 4U

/**
 * @brief Cargo handle
 */
struct cargo_handle
{
	acc_detector_distance_config_t *distance_config;
	acc_detector_distance_handle_t *distance_handle;

	acc_detector_presence_config_t  *presence_config;
	acc_detector_presence_handle_t  *presence_handle;
	acc_detector_presence_metadata_t presence_metadata;

	/* fields in 'parameters' are config-dependent but does not change */
	struct
	{
		cargo_container_size_t container_size;

		uint16_t num_frames_distance_occupies_cargo_app;
		uint16_t num_frames_presence_occupies_cargo_app;
		uint16_t num_frames_presence_uses_same_filter;

	} parameters;

	/* fields in 'state' are config-dependent and changes as fast as every prepare/process */
	struct
	{
		cargo_mode_t current_mode;

		uint16_t num_distance_process_since_prepare;
		uint16_t num_presence_process_since_prepare;
		uint16_t num_presence_process_since_filter_reset;
	} state;
};

//-----------------------------
// Private declarations
//-----------------------------

static bool validate_cargo_config(const cargo_config_t *cargo_config);

static bool populate_handle_with_distance_resources(const cargo_config_t *cargo_config, cargo_handle_t *cargo_handle);

static bool populate_handle_with_presence_resources(const cargo_config_t *cargo_config, cargo_handle_t *cargo_handle);

static bool translate_cargo_config_to_distance_config(const cargo_config_t *cargo_config, acc_detector_distance_config_t *distance_config);

static bool translate_cargo_config_to_presence_config(const cargo_config_t *cargo_config, acc_detector_presence_config_t *presence_config);

static uint16_t presence_calc_num_frames_per_burst(const cargo_config_t *cargo_config);

static float float_clip(float value, float min, float max);

//-----------------------------
// Public definitions
//-----------------------------

void cargo_config_initialize(cargo_config_t *cargo_config, cargo_preset_t preset)
{
	bool status = cargo_config != NULL;

	if (status)
	{
		// >>> base settings
		cargo_config->activate_utilization_level = true;
		cargo_config->activate_presence          = false;

		cargo_config->presence.update_rate               = 6.0f;
		cargo_config->presence.signal_quality            = 20.0f;
		cargo_config->presence.sweeps_per_frame          = 12U;
		cargo_config->presence.inter_detection_threshold = 4.0f;
		cargo_config->presence.intra_detection_threshold = 2.75;
		cargo_config->presence.rate_is_app_driven        = false;

		cargo_config->utilization.signal_quality        = 10.0f;
		cargo_config->utilization.threshold_sensitivity = 0.0f;
		// <<< base settings

		switch (preset)
		{
			case CARGO_PRESET_CONTAINER_10_FEET:
				cargo_config->container_size = CARGO_CONTAINER_SIZE_10_FT;
				break;
			case CARGO_PRESET_CONTAINER_20_FEET:
				cargo_config->container_size = CARGO_CONTAINER_SIZE_20_FT;
				break;
			case CARGO_PRESET_CONTAINER_40_FEET:
				cargo_config->container_size = CARGO_CONTAINER_SIZE_40_FT;
				break;
			case CARGO_PRESET_NO_LENS:
				cargo_config->container_size = CARGO_CONTAINER_SIZE_20_FT;

				cargo_config->utilization.signal_quality        = 25.0f;
				cargo_config->utilization.threshold_sensitivity = 0.5f;

				cargo_config->presence.signal_quality            = 30.0f;
				cargo_config->presence.inter_detection_threshold = 2.0f;
				cargo_config->presence.intra_detection_threshold = 2.0f;
				break;
			default:
				ACC_LOG_ERROR("Invalid preset");
				status = false;
				break;
		}
	}

	if (status)
	{
		cargo_config->magic = CARGO_CONFIG_MAGIC;
	}
}

void cargo_config_log(const cargo_config_t *cargo_config)
{
	switch (cargo_config->container_size)
	{
		case CARGO_CONTAINER_SIZE_10_FT:
			ACC_LOG_INFO("container_size:                     CARGO_CONTAINER_SIZE_10_FT");
			break;
		case CARGO_CONTAINER_SIZE_20_FT:
			ACC_LOG_INFO("container_size:                     CARGO_CONTAINER_SIZE_20_FT");
			break;
		case CARGO_CONTAINER_SIZE_40_FT:
			ACC_LOG_INFO("container_size:                     CARGO_CONTAINER_SIZE_40_FT");
			break;
		default:
			ACC_LOG_INFO("container_size:                     ?");
			break;
	}

	ACC_LOG_INFO("activate_utilization_level:         %s", cargo_config->activate_utilization_level ? "true" : "false");
	ACC_LOG_INFO("utilization.signal_quality:         %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->utilization.signal_quality));
	ACC_LOG_INFO("utilization.threshold_sensitivity:  %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->utilization.threshold_sensitivity));
	ACC_LOG_INFO("activate_presence:                  %s", cargo_config->activate_presence ? "true" : "false");
	ACC_LOG_INFO("presence.update_rate:               %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->presence.update_rate));
	ACC_LOG_INFO("presence.sweeps_per_frame:          %" PRIu16, cargo_config->presence.sweeps_per_frame);
	ACC_LOG_INFO("presence.signal_quality:            %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->presence.signal_quality));
	ACC_LOG_INFO("presence.inter_detection_threshold: %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->presence.inter_detection_threshold));
	ACC_LOG_INFO("presence.intra_detection_threshold: %" PRIfloat, ACC_LOG_FLOAT_TO_INTEGER(cargo_config->presence.intra_detection_threshold));
}

cargo_handle_t *cargo_handle_create(const cargo_config_t *cargo_config)
{
	cargo_handle_t *handle = NULL;
	bool            status = validate_cargo_config(cargo_config);

	if (status)
	{
		handle = acc_integration_mem_calloc(1U, sizeof(*handle));
		status = handle != NULL;
	}

	if (status)
	{
		handle->parameters.container_size                     = cargo_config->container_size;
		handle->state.current_mode                            = CARGO_MODE_NONE;
		handle->state.num_distance_process_since_prepare      = UINT16_MAX;
		handle->state.num_presence_process_since_prepare      = UINT16_MAX;
		handle->state.num_presence_process_since_filter_reset = UINT16_MAX;

		if (cargo_config->activate_presence && cargo_config->activate_utilization_level)
		{
			if (!populate_handle_with_distance_resources(cargo_config, handle) || !populate_handle_with_presence_resources(cargo_config, handle))
			{
				status = false;
			}
			else
			{
				// presence handle is created with "reset" filters:
				handle->state.num_presence_process_since_filter_reset = 0U;

				handle->parameters.num_frames_distance_occupies_cargo_app = 1U;

				uint16_t num_frames_per_burst = presence_calc_num_frames_per_burst(cargo_config);

				handle->parameters.num_frames_presence_occupies_cargo_app = num_frames_per_burst;
				handle->parameters.num_frames_presence_uses_same_filter   = num_frames_per_burst;
			}
		}
		else if (cargo_config->activate_utilization_level)
		{
			if (!populate_handle_with_distance_resources(cargo_config, handle))
			{
				status = false;
			}
		}
		else if (cargo_config->activate_presence)
		{
			if (!populate_handle_with_presence_resources(cargo_config, handle))
			{
				status = false;
			}
			else
			{
				// presence handle is created with "reset" filters:
				handle->state.num_presence_process_since_filter_reset = 0U;

				uint16_t num_frames_per_burst                             = presence_calc_num_frames_per_burst(cargo_config);
				handle->parameters.num_frames_presence_occupies_cargo_app = num_frames_per_burst;
				handle->parameters.num_frames_presence_uses_same_filter   = num_frames_per_burst;
			}
		}
		else
		{
			// This is unreachable (path should be handled in validate_cargo_config)
			status = false;
		}
	}

	if (!status)
	{
		cargo_handle_destroy(handle);
	}

	return status ? handle : NULL;
}

void cargo_handle_destroy(cargo_handle_t *handle)
{
	if (handle != NULL)
	{
		if (handle->distance_config != NULL)
		{
			acc_detector_distance_config_destroy(handle->distance_config);
			handle->distance_config = NULL;
		}

		if (handle->distance_handle != NULL)
		{
			acc_detector_distance_destroy(handle->distance_handle);
			handle->distance_handle = NULL;
		}

		if (handle->presence_config != NULL)
		{
			acc_detector_presence_config_destroy(handle->presence_config);
			handle->presence_config = NULL;
		}

		if (handle->presence_handle != NULL)
		{
			acc_detector_presence_destroy(handle->presence_handle);
			handle->presence_handle = NULL;
		}

		acc_integration_mem_free(handle);
	}
}

bool cargo_get_buffer_size(const cargo_handle_t *handle, uint32_t *buffer_size)
{
	bool status = (handle != NULL) && (buffer_size != NULL);

	uint32_t ignored;
	uint32_t distance_buffer_size = 0U;
	uint32_t presence_buffer_size = 0U;

	if (status && (handle->distance_handle != NULL))
	{
		status = acc_detector_distance_get_sizes(handle->distance_handle, &distance_buffer_size, &ignored);
	}

	if (status && (handle->presence_handle != NULL))
	{
		status = acc_detector_presence_get_buffer_size(handle->presence_handle, &presence_buffer_size);
	}

	if (status)
	{
		*buffer_size = distance_buffer_size > presence_buffer_size ? distance_buffer_size : presence_buffer_size;
	}

	return status;
}

bool cargo_get_distance_cal_result_static_size(const cargo_handle_t *handle, uint32_t *distance_cal_result_static_size)
{
	bool status = (handle != NULL) && (distance_cal_result_static_size != NULL);

	uint32_t ignored;
	uint32_t res = 0U;

	if (status && (handle->distance_handle != NULL))
	{
		status = acc_detector_distance_get_sizes(handle->distance_handle, &ignored, &res);
	}

	if (status)
	{
		*distance_cal_result_static_size = res;
	}

	return status;
}

bool cargo_calibrate(cargo_handle_t                    *handle,
                     acc_sensor_t                      *sensor,
                     const acc_cal_result_t            *sensor_cal_result,
                     void                              *buffer,
                     uint32_t                           buffer_size,
                     uint8_t                           *distance_cal_result_static,
                     uint32_t                           distance_cal_result_static_size,
                     acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                     bool                              *calibration_complete)
{
	bool status = (handle != NULL) && (calibration_complete != NULL);

	if (status)
	{
		if (handle->distance_handle == NULL)
		{
			*calibration_complete = true;
		}
		else
		{
			status = acc_detector_distance_calibrate(sensor,
			                                         handle->distance_handle,
			                                         sensor_cal_result,
			                                         buffer,
			                                         buffer_size,
			                                         distance_cal_result_static,
			                                         distance_cal_result_static_size,
			                                         distance_cal_result_dynamic,
			                                         calibration_complete);
		}
	}

	return status;
}

bool cargo_update_calibration(cargo_handle_t                    *handle,
                              acc_sensor_t                      *sensor,
                              const acc_cal_result_t            *sensor_cal_result,
                              void                              *buffer,
                              uint32_t                           buffer_size,
                              acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                              bool                              *calibration_complete)
{
	bool status = (handle != NULL) && (calibration_complete != NULL);

	if (status)
	{
		if (handle->distance_handle == NULL)
		{
			*calibration_complete = true;
		}
		else
		{
			status = acc_detector_distance_update_calibration(sensor,
			                                                  handle->distance_handle,
			                                                  sensor_cal_result,
			                                                  buffer,
			                                                  buffer_size,
			                                                  distance_cal_result_dynamic,
			                                                  calibration_complete);
		}
	}

	return status;
}

bool cargo_prepare_utilization(cargo_handle_t         *handle,
                               acc_sensor_t           *sensor,
                               const acc_cal_result_t *sensor_cal_result,
                               void                   *buffer,
                               uint32_t                buffer_size)
{
	bool status = (handle != NULL);

	if (status)
	{
		switch (handle->state.current_mode)
		{
			case CARGO_MODE_NONE:
				status =
				    acc_detector_distance_prepare(handle->distance_handle, handle->distance_config, sensor, sensor_cal_result, buffer, buffer_size);
				if (status)
				{
					handle->state.current_mode                       = CARGO_MODE_UTILIZATION;
					handle->state.num_distance_process_since_prepare = 0U;
				}
				break;
			case CARGO_MODE_PRESENCE:
				ACC_LOG_ERROR("Handle is currently in presence-mode. More calls to cargo_process_presence() is expected");
				status = false;
				break;
			case CARGO_MODE_UTILIZATION:
				ACC_LOG_ERROR("Handle has already prepared for utilization level measurement. More calls to cargo_process_utilization() is expected");
				status = false;
				break;
			default:
				ACC_LOG_ERROR("Unknown current mode");
				status = false;
				break;
		}
	}

	return status;
}

bool cargo_prepare_presence(cargo_handle_t         *handle,
                            acc_sensor_t           *sensor,
                            const acc_cal_result_t *sensor_cal_result,
                            void                   *buffer,
                            uint32_t                buffer_size)
{
	bool status = (handle != NULL);

	if (status)
	{
		switch (handle->state.current_mode)
		{
			case CARGO_MODE_NONE:
				status =
				    acc_detector_presence_prepare(handle->presence_handle, handle->presence_config, sensor, sensor_cal_result, buffer, buffer_size);
				if (status)
				{
					handle->state.current_mode                       = CARGO_MODE_PRESENCE;
					handle->state.num_presence_process_since_prepare = 0U;
				}
				break;
			case CARGO_MODE_PRESENCE:
				ACC_LOG_ERROR("Handle has already prepared for presence measurement. More calls to cargo_process_utilization() is expected");
				status = false;
				break;
			case CARGO_MODE_UTILIZATION:
				ACC_LOG_ERROR("Handle is currently in utilization-mode. More calls to cargo_process_utilization() is expected");
				status = false;
				break;
			default:
				ACC_LOG_ERROR("Unknown current mode");
				status = false;
				break;
		}
	}

	return status;
}

bool cargo_process_utilization(cargo_handle_t                    *handle,
                               uint8_t                           *distance_cal_result_static,
                               acc_detector_cal_result_dynamic_t *distance_cal_result_dynamic,
                               void                              *buffer,
                               cargo_result_t                    *cargo_result,
                               bool                              *result_available,
                               acc_detector_distance_result_t    *nullable_distance_result)
{
	bool                           distance_result_available = false;
	acc_detector_distance_result_t local_distance_result     = {0};

	acc_detector_distance_result_t *distance_result = nullable_distance_result != NULL ? nullable_distance_result : &local_distance_result;

	bool status = (handle != NULL) && (cargo_result != NULL) && (result_available != NULL);

	if (status)
	{
		cargo_result->utilization_valid = false;
		cargo_result->presence_valid    = false;

		switch (handle->state.current_mode)
		{
			case CARGO_MODE_UTILIZATION:
			{
				if (acc_detector_distance_process(handle->distance_handle,
				                                  buffer,
				                                  distance_cal_result_static,
				                                  distance_cal_result_dynamic,
				                                  &distance_result_available,
				                                  distance_result))
				{
					handle->state.num_distance_process_since_prepare++;

					*result_available = distance_result_available;

					if (distance_result->num_distances > 0U)
					{
						float distance_m = distance_result->distances[0];

						float container_size_length_m = 0.0f;
						switch (handle->parameters.container_size)
						{
							case CARGO_CONTAINER_SIZE_10_FT:
								container_size_length_m = 3.0f;
								break;
							case CARGO_CONTAINER_SIZE_20_FT:
								container_size_length_m = 6.0f;
								break;
							case CARGO_CONTAINER_SIZE_40_FT:
								container_size_length_m = 12.0f;
								break;
							default:
								ACC_LOG_ERROR("Unexpected container_size_t");
								status = false;
								break;
						}

						cargo_result->utilization_valid = true;

						cargo_result->distance      = distance_m;
						cargo_result->level_m       = float_clip(container_size_length_m - distance_m, 0.0f, container_size_length_m);
						cargo_result->level_percent = float_clip(cargo_result->level_m / container_size_length_m * 100.0f, 0.0f, 100.0f);

						cargo_result->calibration_needed = distance_result->processing_result->calibration_needed;
					}
				}
				else
				{
					ACC_LOG_ERROR("distance detector processing failed");
					status = false;
				}

				if (handle->state.num_distance_process_since_prepare >= handle->parameters.num_frames_distance_occupies_cargo_app)
				{
					handle->state.current_mode = CARGO_MODE_NONE;
				}
			}
			break;
			case CARGO_MODE_PRESENCE:
				ACC_LOG_ERROR("Handle is in presence-mode. Call cargo_process_presence() instead");
				status = false;
				break;
			case CARGO_MODE_NONE:
				ACC_LOG_ERROR("cargo_prepare_utilization() have not been called beforehand");
				status = false;
				break;
			default:
				ACC_LOG_ERROR("Unknown current mode");
				status = false;
				break;
		}
	}

	return status;
}

bool cargo_process_presence(cargo_handle_t                 *handle,
                            void                           *buffer,
                            cargo_result_t                 *cargo_result,
                            acc_detector_presence_result_t *nullable_presence_result)
{

	acc_detector_presence_result_t  local_presence_result = {0};
	acc_detector_presence_result_t *presence_result       = nullable_presence_result != NULL ? nullable_presence_result : &local_presence_result;

	bool status = (handle != NULL) && (cargo_result != NULL);

	if (status)
	{
		cargo_result->utilization_valid = false;
		cargo_result->presence_valid    = false;

		switch (handle->state.current_mode)
		{
			case CARGO_MODE_PRESENCE:
			{
				status = acc_detector_presence_process(handle->presence_handle, buffer, presence_result);
				if (status)
				{
					handle->state.num_presence_process_since_prepare++;
					handle->state.num_presence_process_since_filter_reset++;

					cargo_result->presence_valid = true;

					cargo_result->presence_detected    = presence_result->presence_detected;
					cargo_result->inter_presence_score = presence_result->inter_presence_score;
					cargo_result->intra_presence_score = presence_result->intra_presence_score;
					cargo_result->calibration_needed   = presence_result->processing_result.calibration_needed;

					if (handle->state.num_presence_process_since_filter_reset >= handle->parameters.num_frames_presence_uses_same_filter)
					{
						acc_detector_presence_reset_filters(handle->presence_handle);
						handle->state.num_presence_process_since_filter_reset = 0U;
					}

					if (handle->state.num_presence_process_since_prepare >= handle->parameters.num_frames_presence_occupies_cargo_app)
					{
						handle->state.current_mode = CARGO_MODE_NONE;
					}
				}
			}
			break;
			case CARGO_MODE_UTILIZATION:
				ACC_LOG_ERROR("Handle is in utilization-mode. Call cargo_process_utilization() instead");
				status = false;
				break;
			case CARGO_MODE_NONE:
				ACC_LOG_ERROR("cargo_prepare_presence() have not been called beforehand");
				status = false;
				break;
			default:
				ACC_LOG_ERROR("Unknown current mode");
				status = false;
				break;
		}
	}

	return status;
}

cargo_mode_t cargo_current_mode_get(const cargo_handle_t *handle)
{
	cargo_mode_t res = CARGO_MODE_NONE;

	if (handle != NULL)
	{
		res = handle->state.current_mode;
	}

	return res;
}

bool cargo_presence_metadata_get(cargo_handle_t *handle, acc_detector_presence_metadata_t *presence_metadata)
{
	bool status = (handle != NULL) && (presence_metadata != NULL);

	if (status)
	{
		status = handle->presence_handle != NULL;
	}

	if (status)
	{
		*presence_metadata = handle->presence_metadata;
	}

	return status;
}

//-----------------------------
// Private definitions
//-----------------------------

static bool validate_cargo_config(const cargo_config_t *cargo_config)
{
	bool status = true;

	if (cargo_config == NULL)
	{
		ACC_LOG_ERROR("Cargo config is NULL");
		status = false;
	}

	if (status && (cargo_config->magic != CARGO_CONFIG_MAGIC))
	{
		ACC_LOG_ERROR("Cargo config has not been initialized");
		status = false;
	}

	if (status && cargo_config->activate_presence)
	{

		if (status && (cargo_config->presence.update_rate < CARGO_PRESENCE_MIN_UPDATE_RATE))
		{
			ACC_LOG_ERROR("Presence update_rate is too low.");
			status = false;
		}

		if (status && (cargo_config->presence.sweeps_per_frame < CARGO_PRESENCE_MIN_SWEEPS_PER_FRAME))
		{
			ACC_LOG_ERROR("Presence sweeps_per_frame is too low.");
			status = false;
		}
	}

	if (status && (!cargo_config->activate_utilization_level && !cargo_config->activate_presence))
	{
		ACC_LOG_ERROR("Need to activate at least utilization or presence");
		status = false;
	}

	return status;
}

static bool populate_handle_with_distance_resources(const cargo_config_t *cargo_config, cargo_handle_t *cargo_handle)
{
	bool status = true;

	if (status)
	{
		cargo_handle->distance_config = acc_detector_distance_config_create();
		status                        = cargo_handle->distance_config != NULL;
	}

	if (status)
	{
		status = translate_cargo_config_to_distance_config(cargo_config, cargo_handle->distance_config);
	}

	if (status)
	{
		cargo_handle->distance_handle = acc_detector_distance_create(cargo_handle->distance_config);
		status                        = cargo_handle->distance_handle != NULL;
	}

	return status;
}

static bool populate_handle_with_presence_resources(const cargo_config_t *cargo_config, cargo_handle_t *cargo_handle)
{
	bool status = true;

	if (status)
	{
		cargo_handle->presence_config = acc_detector_presence_config_create();
		status                        = cargo_handle->presence_config != NULL;
	}

	if (status)
	{
		status = translate_cargo_config_to_presence_config(cargo_config, cargo_handle->presence_config);
	}

	if (status)
	{
		cargo_handle->presence_handle = acc_detector_presence_create(cargo_handle->presence_config, &cargo_handle->presence_metadata);
		status                        = cargo_handle->presence_handle != NULL;
	}

	return status;
}

static bool translate_cargo_config_to_distance_config(const cargo_config_t *cargo_config, acc_detector_distance_config_t *distance_config)
{
	bool status = true;

	const float start_m = 0.3;
	float       end_m;

	switch (cargo_config->container_size)
	{
		case CARGO_CONTAINER_SIZE_10_FT:
			end_m = 3.5;
			break;
		case CARGO_CONTAINER_SIZE_20_FT:
			end_m = 6.5;
			break;
		case CARGO_CONTAINER_SIZE_40_FT:
			end_m = 12.5;
			break;
		default:
			ACC_LOG_ERROR("Unexpected container size in config");
			status = false;
	}

	if (status)
	{
		acc_detector_distance_config_start_set(distance_config, start_m);
		acc_detector_distance_config_end_set(distance_config, end_m);

		acc_detector_distance_config_signal_quality_set(distance_config, cargo_config->utilization.signal_quality);
		acc_detector_distance_config_threshold_sensitivity_set(distance_config, cargo_config->utilization.threshold_sensitivity);
		acc_detector_distance_config_reflector_shape_set(distance_config, ACC_DETECTOR_DISTANCE_REFLECTOR_SHAPE_PLANAR);
	}

	return status;
}

static bool translate_cargo_config_to_presence_config(const cargo_config_t *cargo_config, acc_detector_presence_config_t *presence_config)
{
	bool status = true;

	const float start_m = 0.8f;
	float       end_m;

	switch (cargo_config->container_size)
	{
		case CARGO_CONTAINER_SIZE_10_FT:
			end_m = 2.7;
			break;
		case CARGO_CONTAINER_SIZE_20_FT:
			end_m = 5.7;
			break;
		case CARGO_CONTAINER_SIZE_40_FT:
			end_m = 7.0;
			break;
		default:
			ACC_LOG_ERROR("Unexpected container size in config");
			status = false;
	}

	if (status)
	{
		acc_detector_presence_config_start_set(presence_config, start_m);
		acc_detector_presence_config_end_set(presence_config, end_m);

		acc_detector_presence_config_frame_rate_app_driven_set(presence_config, cargo_config->presence.rate_is_app_driven);

		acc_detector_presence_config_intra_detection_threshold_set(presence_config, cargo_config->presence.intra_detection_threshold);
		acc_detector_presence_config_intra_frame_time_const_set(presence_config, 0.15f);
		acc_detector_presence_config_intra_output_time_const_set(presence_config, 0.30f);
		acc_detector_presence_config_inter_detection_threshold_set(presence_config, cargo_config->presence.inter_detection_threshold);
		acc_detector_presence_config_inter_frame_slow_cutoff_set(presence_config, 0.20f);
		acc_detector_presence_config_inter_frame_fast_cutoff_set(presence_config, cargo_config->presence.update_rate);
		acc_detector_presence_config_inter_frame_deviation_time_const_set(presence_config, 0.50f);
		acc_detector_presence_config_inter_output_time_const_set(presence_config, 2.00f);
		acc_detector_presence_config_inter_frame_presence_timeout_set(presence_config, 0U); // No timeout

		acc_detector_presence_config_automatic_subsweeps_set(presence_config, true);
		acc_detector_presence_config_sweeps_per_frame_set(presence_config, cargo_config->presence.sweeps_per_frame);
		acc_detector_presence_config_signal_quality_set(presence_config, cargo_config->presence.signal_quality);

		acc_detector_presence_config_frame_rate_set(presence_config, cargo_config->presence.update_rate);
	}

	return status;
}

static uint16_t presence_calc_num_frames_per_burst(const cargo_config_t *cargo_config)
{
	// "+0.5f" will round (instead of truncate) when casting to uint16_t
	float num_frames_per_burst_f = (CARGO_PRESENCE_RUN_TIME_S * cargo_config->presence.update_rate) + 0.5f;
	return (uint16_t)num_frames_per_burst_f;
}

static float float_clip(float value, float min, float max)
{
	float res;

	if (value < min)
	{
		res = min;
	}
	else if (value > max)
	{
		res = max;
	}
	else
	{
		res = value;
	}

	return res;
}
