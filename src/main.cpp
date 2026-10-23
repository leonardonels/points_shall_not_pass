#include "points_shall_not_pass/filter.hpp"

int main(int argc, char** argv) {
    // rclcpp::init installs its own SIGINT handler: Ctrl-C makes spin() return.
    rclcpp::init(argc, argv);

    auto node = std::make_shared<FilterNode>();

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
