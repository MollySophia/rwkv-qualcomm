#pragma once

enum platform_type {
    PLATFORM_SNAPDRAGON, // lets add snapdragon support first
    PLATFORM_UNKNOWN,
};

struct snapdragon_soc_id {
    int soc_id;
    const char * soc_partname;
    const char * soc_name;
    const char * htp_arch;
};

class soc_detect {
    public:
        soc_detect();
        ~soc_detect();

        int detect_platform();

        platform_type get_platform_type();
        const char * get_platform_name();
        const char * get_soc_name();
        const char * get_soc_partname();
        const char * get_htp_arch();
    private:
        platform_type m_platform_type = PLATFORM_UNKNOWN;
        int m_soc_id = 0;
        const char * m_soc_name = "Unknown";
        const char * m_soc_partname = "Unknown";
        const char * m_htp_arch = "Unknown";
};
