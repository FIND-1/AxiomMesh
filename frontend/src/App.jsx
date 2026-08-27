import { useCallback, useEffect, useRef, useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCreatingConversation, setIsCreatingConversation] = useState(false);
  const currentConversationIdRef = useRef(null);
  const activeStreamIdsRef = useRef(new Set());

  const loadConversations = useCallback(async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  }, []);

  useEffect(() => {
    currentConversationIdRef.current = currentConversationId;
  }, [currentConversationId]);

  // Load conversations on mount
  useEffect(() => {
    let isMounted = true;

    api.listConversations()
      .then((convs) => {
        if (isMounted) {
          setConversations(convs);
        }
      })
      .catch((error) => {
        console.error('Failed to load conversations:', error);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Load conversation details when selected
  useEffect(() => {
    if (!currentConversationId) {
      return undefined;
    }

    let isMounted = true;
    api.getConversation(currentConversationId)
      .then((conv) => {
        if (isMounted) {
          setCurrentConversation(conv);
        }
      })
      .catch((error) => {
        console.error('Failed to load conversation:', error);
      });

    return () => {
      isMounted = false;
    };
  }, [currentConversationId]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState !== 'visible') {
        return;
      }

      loadConversations();
      const selectedConversationId = currentConversationIdRef.current;
      if (
        selectedConversationId &&
        !activeStreamIdsRef.current.has(selectedConversationId)
      ) {
        api.getConversation(selectedConversationId)
          .then((conv) => {
            if (currentConversationIdRef.current === selectedConversationId) {
              setCurrentConversation(conv);
            }
          })
          .catch((error) => {
            console.error('Failed to refresh conversation:', error);
          });
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [loadConversations]);

  const createStreamingAssistantMessage = () => ({
    role: 'assistant',
    stage1: null,
    stage2: null,
    stage3: null,
    metadata: null,
    agent_events: [],
    is_streaming: true,
    loading: {
      stage1: false,
      stage2: false,
      stage3: false,
    },
  });

  const updateStreamingAssistant = useCallback((conversationId, updateMessage) => {
    setCurrentConversation((prev) => {
      if (!prev || prev.id !== conversationId) {
        return prev;
      }

      const messages = [...prev.messages];
      const lastMessage = messages[messages.length - 1];
      const assistantMessage =
        lastMessage?.role === 'assistant' && lastMessage.is_streaming
          ? {
              ...lastMessage,
              loading: { ...(lastMessage.loading || {}) },
              agent_events: [...(lastMessage.agent_events || [])],
            }
          : createStreamingAssistantMessage();

      updateMessage(assistantMessage);

      if (lastMessage?.role === 'assistant' && lastMessage.is_streaming) {
        messages[messages.length - 1] = assistantMessage;
      } else {
        messages.push(assistantMessage);
      }

      return { ...prev, messages };
    });
  }, []);

  const handleNewConversation = async () => {
    if (isCreatingConversation) {
      return;
    }

    if (currentConversation?.messages?.length === 0) {
      return;
    }

    const emptyConversation = conversations.find((conv) => conv.message_count === 0);
    if (emptyConversation) {
      setCurrentConversationId(emptyConversation.id);
      return;
    }

    setIsCreatingConversation(true);
    try {
      const newConv = await api.createConversation();
      setConversations((prev) => {
        if (prev.some((conv) => conv.id === newConv.id)) {
          return prev;
        }

        return [
          {
            id: newConv.id,
            created_at: newConv.created_at,
            title: newConv.title,
            message_count: 0,
          },
          ...prev,
        ];
      });
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    } finally {
      setIsCreatingConversation(false);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  const handleSendMessage = async (content) => {
    const conversationId = currentConversationId;
    if (!conversationId) return;

    setIsLoading(true);
    activeStreamIdsRef.current.add(conversationId);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => {
        if (!prev || prev.id !== conversationId) {
          return prev;
        }

        return {
          ...prev,
          messages: [...prev.messages, userMessage],
        };
      });

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = createStreamingAssistantMessage();

      // Add the partial assistant message
      setCurrentConversation((prev) => {
        if (!prev || prev.id !== conversationId) {
          return prev;
        }

        return {
          ...prev,
          messages: [...prev.messages, assistantMessage],
        };
      });

      // Send message with streaming
      await api.sendMessageStream(conversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            updateStreamingAssistant(conversationId, (message) => {
              message.loading.stage1 = true;
            });
            break;

          case 'agent_status':
            updateStreamingAssistant(conversationId, (message) => {
              message.agent_events = [...(message.agent_events || []), event];
            });
            break;

          case 'stage1_complete':
            updateStreamingAssistant(conversationId, (message) => {
              message.stage1 = event.data;
              message.loading.stage1 = false;
            });
            break;

          case 'stage2_start':
            updateStreamingAssistant(conversationId, (message) => {
              message.loading.stage2 = true;
            });
            break;

          case 'stage2_complete':
            updateStreamingAssistant(conversationId, (message) => {
              message.stage2 = event.data;
              message.metadata = event.metadata;
              message.loading.stage2 = false;
            });
            break;

          case 'stage3_start':
            updateStreamingAssistant(conversationId, (message) => {
              message.loading.stage3 = true;
            });
            break;

          case 'stage3_complete':
            updateStreamingAssistant(conversationId, (message) => {
              message.stage3 = event.data;
              message.loading.stage3 = false;
            });
            break;

          case 'title_complete':
            setCurrentConversation((prev) =>
              prev?.id === conversationId
                ? { ...prev, title: event.data.title }
                : prev
            );
            setConversations((prev) =>
              prev.map((conv) =>
                conv.id === conversationId
                  ? { ...conv, title: event.data.title }
                  : conv
              )
            );
            loadConversations();
            break;

          case 'complete':
            // Stream complete, reload conversations list
            loadConversations();
            activeStreamIdsRef.current.delete(conversationId);
            updateStreamingAssistant(conversationId, (message) => {
              message.is_streaming = false;
            });
            if (currentConversationIdRef.current === conversationId) {
              api.getConversation(conversationId)
                .then((conv) => {
                  if (currentConversationIdRef.current === conversationId) {
                    setCurrentConversation(conv);
                  }
                })
                .catch((error) => {
                  console.error('Failed to reload completed conversation:', error);
                });
            }
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      activeStreamIdsRef.current.delete(conversationId);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => {
        if (!prev || prev.id !== conversationId) {
          return prev;
        }

        return {
          ...prev,
          messages: prev.messages.slice(0, -2),
        };
      });
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        isCreatingConversation={isCreatingConversation}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;
