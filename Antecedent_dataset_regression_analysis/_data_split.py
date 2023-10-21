def main():
    with open('top_thres0.3_viral_checked.txt', 'r') as file:
        lines = file.readlines()

    # Split lines into four groups
    group_1 = lines[:556]
    group_2 = lines[556:1112]
    group_3 = lines[1112:1669]
    group_4 = lines[1669:]

    # Write groups to separate files
    write_group_to_file(group_1, 'top_thres0.3_viral_checked_Alan.txt')
    write_group_to_file(group_2, 'top_thres0.3_viral_checked_Alex.txt')
    write_group_to_file(group_3, 'top_thres0.3_viral_checked_Ken.txt')
    write_group_to_file(group_4, 'top_thres0.3_viral_checked_Qiaosong.txt')

def write_group_to_file(group, filename):
    with open(filename, 'a') as file:
        for line in group:
            file.write(line)

main()
