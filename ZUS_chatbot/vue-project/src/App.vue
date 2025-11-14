<template>
  <div id="chat-app">
    <header class="app-header">
      <h1>ZUS Coffee Assistant ☕</h1>
      <div class="session-id">Session: {{ sessionId }}</div>
    </header>

    <div class="chat-container" ref="chatContainer">
      <div
        v-for="message in messages"
        :key="message.id"
        class="message-wrapper"
        :class="message.sender"
      >
        <div class="avatar">
          {{ message.sender === "user" ? "You" : "AI" }}
        </div>
        
        <div class="message-bubble-wrapper">
          <div class="message-bubble">
            <div v-html="formattedText(message.text)"></div>
          </div>
          <div class="timestamp">
            {{ formatTimestamp(message.timestamp) }}
          </div>
        </div>
      </div>
      
      <div v-if="isLoading" class="message-wrapper ai">
        <div class="avatar">AI</div>
        <div class="message-bubble-wrapper">
          <div class="message-bubble">
            <div class="typing-indicator">
              <span></span><span></span><span></span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="showAutocomplete" class="autocomplete-popup">
      <div 
        v-for="cmd in filteredCommands" 
        :key="cmd.command" 
        class="autocomplete-item"
        @click="selectCommand(cmd)"
      >
        <strong>{{ cmd.command }}</strong> - <span>{{ cmd.desc }}</span>
      </div>
    </div>

    <footer class="input-area">
      <form @submit.prevent="sendMessage">
        <textarea
          ref="textareaRef"
          v-model="newMessage"
          placeholder="Type / for commands or ask a question..."
          :disabled="isLoading"
          rows="1"
          @keydown.enter="handleEnter"
          @input="autoResizeTextarea"
        ></textarea>
        <button type="submit" :disabled="isLoading || !newMessage.trim()">
          Send
        </button>
      </form>
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from "vue";

// --- Configuration ---
const AGENT_API_URL = "http://127.0.0.1:8001/chat";
const LOCAL_STORAGE_MESSAGES_KEY = "chat_messages";
const LOCAL_STORAGE_SESSION_KEY = "chat_session_id";

// NEW: Define the welcome message text
const WELCOME_MESSAGE_TEXT = `Hello! I'm the ZUS Coffee Assistant.

You can ask me questions directly, or use these commands:
* /products [query] - Search for ZUS products
* /outlets [query] - Find store locations or hours
* /calculator [expression] - Evaluate a math expression
* /reset - Start a new conversation`;

// --- Command Definitions ---
const slashCommands = ref([
  { command: "/products", desc: "Search for ZUS products" },
  { command: "/outlets", desc: "Find store locations or hours" },
  { command: "/calculator", desc: "Evaluate a math expression" },
  { command: "/reset", desc: "Start a new conversation" },
]);

// --- State Refs ---
const messages = ref([]);
const newMessage = ref("");
const isLoading = ref(false);
const sessionId = ref("");
const chatContainer = ref(null); // For scrolling
const textareaRef = ref(null); // For focus and resizing
const showAutocomplete = ref(false);
const filteredCommands = ref([]);

// --- Lifecycle & Persistence ---

onMounted(() => {
  // 1. Load Session ID
  const storedSessionId = localStorage.getItem(LOCAL_STORAGE_SESSION_KEY);
  if (storedSessionId) {
    sessionId.value = storedSessionId;
  } else {
    sessionId.value = generateSessionId();
    localStorage.setItem(LOCAL_STORAGE_SESSION_KEY, sessionId.value);
  }

  // 2. Load Messages
const storedMessages = localStorage.getItem(LOCAL_STORAGE_MESSAGES_KEY);
if (storedMessages) {
  messages.value = JSON.parse(storedMessages);
} else {
  // Add welcome message if no history
  addMessage("ai", WELCOME_MESSAGE_TEXT); // <-- USE THE NEW CONSTANT
}

  // 3. Scroll to bottom on load
  scrollToBottom();
});

// Watch for changes and save to localStorage
watch(
  messages,
  (newMessages) => {
    localStorage.setItem(LOCAL_STORAGE_MESSAGES_KEY, JSON.stringify(newMessages));
  },
  { deep: true } // Necessary for watching changes inside the array
);

// --- Core Chat Logic ---

async function sendMessage() {
  const userText = newMessage.value.trim();
  if (userText === "" || isLoading.value) return;

  // Clear input and autocomplete
  newMessage.value = "";
  showAutocomplete.value = false;
  autoResizeTextarea(); // Reset textarea height

  // Handle client-side /reset command
  if (userText === "/reset") {
    addMessage("user", "/reset");
    resetChat();
    return;
  }

  // 1. Add user message to UI
  addMessage("user", userText);
  isLoading.value = true;
  await scrollToBottom();

  try {
    // 2. Call the Agent API
    const response = await fetch(AGENT_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: userText,
        session_id: sessionId.value,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        errorData.detail || `HTTP error! Status: ${response.status}`
      );
    }
    const data = await response.json();
    
    // 3. Add AI response to UI
    addMessage("ai", data.answer);
    
  } catch (error) {
    console.error("Error sending message:", error);
    // 4. Add error message to UI
    addMessage("ai", `Sorry, I ran into an error: ${error.message}`);
    
  } finally {
    // 5. Reset loading state and scroll
    isLoading.value = false;
    await scrollToBottom();
  }
}

/**
 * Resets the chat state and starts a new session
 */
function resetChat() {
  console.log("Resetting chat...");
  // Clear messages
  messages.value = [];

  // Generate new session ID
  sessionId.value = generateSessionId();
  localStorage.setItem(LOCAL_STORAGE_SESSION_KEY, sessionId.value);

  // Add new welcome message
  addMessage("ai", WELCOME_MESSAGE_TEXT); // <-- USE THE NEW CONSTANT

  // Clear local storage for messages
  localStorage.removeItem(LOCAL_STORAGE_MESSAGES_KEY);
}

// --- Input & Composer Handlers ---

/**
 * Handles Enter key presses for sending or newlines
 */
function handleEnter(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    // Send message on Enter
    event.preventDefault();
    sendMessage();
  }
  // Allow newline on Shift+Enter (default behavior)
}

/**
 * Dynamically resizes the textarea based on content
 */
function autoResizeTextarea() {
  if (textareaRef.value) {
    textareaRef.value.style.height = "auto"; // Reset height
    textareaRef.value.style.height = `${textareaRef.value.scrollHeight}px`; // Set to scroll height
  }
}

// --- Autocomplete Handlers ---

// Watch the input for slash commands
watch(newMessage, (value) => {
  if (value.startsWith("/") && !value.includes(" ")) {
    const query = value.substring(1).toLowerCase();
    filteredCommands.value = slashCommands.value.filter((cmd) =>
      cmd.command.toLowerCase().includes(query)
    );
    showAutocomplete.value = filteredCommands.value.length > 0;
  } else {
    showAutocomplete.value = false;
  }
  
  // Also auto-resize as user types
  autoResizeTextarea();
});

/**
 * Selects a command from the autocomplete list
 */
function selectCommand(command) {
  newMessage.value = command.command + " ";
  showAutocomplete.value = false;
  textareaRef.value?.focus(); // Focus the input after selection
}

// --- Utility Functions ---

/**
 * Helper to add a new message to the state
 */
function addMessage(sender, text) {
  messages.value.push({
    id: Date.now() + Math.random(), // Unique ID
    sender: sender, // 'user' or 'ai'
    text: text,
    timestamp: new Date().toISOString(), // Use ISO string for persistence
  });
}

/**
* Generates a simple unique session ID
*/
function generateSessionId() {
  return "vue-session-" + Math.random().toString(36).substring(2, 11);
}

/**
* Scrolls the chat container to the bottom
*/
async function scrollToBottom() {
  await nextTick();
  const container = chatContainer.value;
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

/**
* Formats text to replace newlines with <br> tags
*/
function formattedText(text) {
  if (typeof text !== 'string') return '';
  return text.replace(/\n/g, "<br>");
}

/**
 * Formats an ISO timestamp to a readable time
 */
function formatTimestamp(isoString) {
  if (!isoString) return '';
  return new Date(isoString).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}
</script>

<style>
/* Basic App Styling */
:root {
  --blue: #007aff;
  --grey-light: #f0f0f0;
  --grey-dark: #8e8e93;
  --border-color: #e0e0e0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif;
  margin: 0;
  padding: 0;
  background-color: #f9f9f9;
}

#chat-app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  max-width: 768px;
  margin: 0 auto;
  background-color: #ffffff;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
  border-left: 1px solid var(--border-color);
  border-right: 1px solid var(--border-color);
}

/* Header */
.app-header {
  background-color: #fff;
  border-bottom: 1px solid var(--border-color);
  padding: 1rem;
  text-align: center;
  position: relative;
  flex-shrink: 0;
}
.app-header h1 { margin: 0; font-size: 1.25rem; color: #333; }
.session-id { font-size: 0.75rem; color: var(--grey-dark); margin-top: 4px; }

/* Chat Container */
.chat-container {
  flex-grow: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem; /* Increased gap for avatars */
}

/* NEW: Message Wrapper */
.message-wrapper {
  display: flex;
  align-items: flex-start; /* Align avatar and bubble */
  gap: 0.5rem;
  width: 100%;
}
.message-bubble-wrapper {
  max-width: 85%;
  display: flex;
  flex-direction: column;
}
.message-bubble {
  max-width: 100%;
  padding: 0.75rem 1rem;
  border-radius: 18px;
  line-height: 1.4;
  word-wrap: break-word;
}
.timestamp {
  font-size: 0.75rem;
  color: var(--grey-dark);
  margin-top: 4px;
}

/* NEW: Avatar */
.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: var(--grey-light);
  color: #555;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.9rem;
  flex-shrink: 0;
}

/* AI Messages */
.message-wrapper.ai {
  justify-content: flex-start;
}
.message-wrapper.ai .message-bubble {
  background-color: var(--grey-light);
  color: #000;
  border-top-left-radius: 4px;
}
.message-wrapper.ai .timestamp {
  align-self: flex-start;
  margin-left: 4px;
}
.message-wrapper.ai .avatar {
  background-color: #dbeafe; /* Light blue for AI */
  color: #1e40af;
}

/* User Messages */
.message-wrapper.user {
  flex-direction: row-reverse; /* Flip order for user */
}
.message-wrapper.user .message-bubble-wrapper {
  align-items: flex-end;
}
.message-wrapper.user .message-bubble {
  background-color: var(--blue);
  color: #fff;
  border-top-right-radius: 4px;
}
.message-wrapper.user .timestamp {
  align-self: flex-end;
  margin-right: 4px;
}
.message-wrapper.user .avatar {
  background-color: #e0f2fe; /* Lighter blue for User */
  color: #0c4a6e;
}

/* Input Area */
.input-area {
  padding: 1rem;
  background-color: #fff;
  border-top: 1px solid var(--border-color);
  flex-shrink: 0;
  position: relative; /* For autocomplete */
}

.input-area form {
  display: flex;
  gap: 0.5rem;
  align-items: flex-end; /* Align button with textarea */
}

/* NEW: Textarea */
.input-area textarea {
  flex-grow: 1;
  padding: 0.75rem 1rem;
  border: 1px solid var(--border-color);
  border-radius: 20px;
  font-size: 1rem;
  font-family: inherit;
  line-height: 1.4;
  resize: none; /* Disable manual resize */
  max-height: 150px; /* Limit height */
  overflow-y: auto; /* Add scroll if it gets too tall */
}

.input-area button {
  padding: 0.75rem 1.25rem;
  background-color: var(--blue);
  color: white;
  border: none;
  border-radius: 20px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  height: 44px; /* Match default textarea height */
  flex-shrink: 0;
}
.input-area button:disabled {
  background-color: var(--grey-dark);
  cursor: not-allowed;
}

/* NEW: Autocomplete */
.autocomplete-popup {
  position: absolute;
  bottom: 100%; /* Position above the input area */
  left: 0;
  right: 0;
  margin: 0 1rem 0.5rem 1rem; /* Match input padding */
  background: white;
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.08);
  max-height: 200px;
  overflow-y: auto;
  z-index: 10;
}
.autocomplete-item {
  padding: 0.75rem 1rem;
  cursor: pointer;
  border-bottom: 1px solid var(--border-color);
}
.autocomplete-item:last-child {
  border-bottom: none;
}
.autocomplete-item:hover {
  background-color: var(--grey-light);
}
.autocomplete-item strong {
  color: var(--blue);
}
.autocomplete-item span {
  font-size: 0.9rem;
  color: var(--grey-dark);
}

/* Typing Indicator */
.typing-indicator span {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  background-color: var(--grey-dark); margin: 0 2px;
  animation: bounce 1.4s infinite both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce { 0%, 80%, 100% { transform: scale(0); } 40% { transform: scale(1); } }
</style>