import { useEffect, type ReactElement } from "react";
import { AppShell } from "@/components/AppShell";
import { NavKey, useAdmin } from "@/store/admin";
import PricesPage from "@/pages/PricesPage";
import ModelsPage from "@/pages/ModelsPage";
import MappingsPage from "@/pages/MappingsPage";
import StatsPage from "@/pages/StatsPage";
import DeepSeekWebPage from "@/pages/DeepSeekWebPage";

export default function App() {
  const { nav, fetchAll } = useAdmin();

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const page: Record<NavKey, ReactElement> = {
    stats: <StatsPage />,
    models: <ModelsPage />,
    mappings: <MappingsPage />,
    prices: <PricesPage />,
    deepseek: <DeepSeekWebPage />,
  };

  return <AppShell>{page[nav]}</AppShell>;
}
