def process_ansi_output(input_text):
    """
    Process text containing ANSI escape codes, carriage returns, and backspace
    to show what would actually be displayed on the terminal.
    \r moves cursor to start of line
    \x1b[K erases from cursor to end of line
    \x08 moves cursor back one position and deletes the previous character
    """
    current_line = []  # List of characters for easy position manipulation
    final_lines = []
    cursor_pos = 0
    i = 0

    while i < len(input_text):
        if input_text[i] == '\r':
            # Carriage return - move cursor back to start of line
            cursor_pos = 0
            i += 1
        elif input_text[i] == '\x08':
            # Backspace - move cursor back and delete previous character
            if cursor_pos > 0:
                cursor_pos -= 1
                if cursor_pos < len(current_line):
                    current_line.pop(cursor_pos)
            i += 1
        elif input_text[i] == '\x1b' and i + 2 < len(input_text) and input_text[i+1] == '[' and input_text[i+2] == 'K':
            # ESC[K - Erase from cursor to end of line
            current_line = current_line[:cursor_pos]
            i += 3
        elif input_text[i] == '\n':
            # Newline - store current line and start a new one
            final_lines.append(''.join(current_line))
            current_line = []
            cursor_pos = 0
            i += 1
        else:
            # Regular character - add/overlay at cursor position
            if cursor_pos >= len(current_line):
                current_line.append(input_text[i])
            else:
                current_line[cursor_pos] = input_text[i]
            cursor_pos += 1
            i += 1

    # Add the last line if it exists
    if current_line:
        final_lines.append(''.join(current_line))

    return "\n".join(final_lines)

